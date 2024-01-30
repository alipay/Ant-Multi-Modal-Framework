# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from antmmf.utils.distributed_utils import gather_tensor, all_gather
from antmmf.utils.distributed_utils import get_world_size, get_rank
from antmmf.utils.general import get_package_version
from .moco_utils import MocoUtils
from .univl_video_base import UnivlVideoBase
from .dmae_utils import DmaeUtils, CrossEn, NegNCE



class UnivlForVideoTextRetrieval(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        with_cross_encoder = "stage2" in self.config.training_stage
        self.module = UnivlVideoBase(config, with_cross_encoder=with_cross_encoder)
        if with_cross_encoder:
            self.dropout = nn.Dropout(0.1)
            self.similarity_dense = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
                nn.ReLU(True),
                nn.Linear(self.config.hidden_size * 2, 1),
            )
        self.with_moco = getattr(self.config, "with_moco", True)
        if self.with_moco:
            self.moco_utils = None

        self.dmae_utils = DmaeUtils(config)
        self.loss_type = getattr(self.config, "l3_loss_type", "negNCE")
        assert self.loss_type in ["cross_entropy", "negNCE"]
        if self.loss_type == "negNCE":
            self.loss_fct = NegNCE()
        else:
            self.loss_fct = CrossEn()

    def _cross_similarity(
        self, sequence_output, visual_output, attention_mask, video_mask, num_clips
    ):
        # 构造文本和视频的不同输入组合，并获取相似度
        # refer to: https://github.com/microsoft/UniVL/blob/main/modules/modeling.py#L341
        b_text, s_text, h_text = sequence_output.size()  # bsz, seq, hidden
        b_visual, s_visual, h_visual = visual_output.size()  # bsz*n_clips, seq, hidden

        retrieve_logits_list = []
        step_size = 5

        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]  # split_size, seq, hidden
            attention_mask_row = attention_mask_splits[i]  # split_size, seq
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(
                1, b_visual, 1, 1
            )  # split_size, b_visual, seq, hidden
            sequence_output_l = sequence_output_l.view(
                -1, s_text, h_text
            )  # split_size*b_visual, seq, hidden
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(
                1, b_visual, 1
            )  # split_size, b_visual, seq
            attention_mask_l = attention_mask_l.view(-1, s_text)  # split_size*bsz, seq

            step_truth = sequence_output_row.size(0)  # split_size
            visual_output_r = visual_output.unsqueeze(0).repeat(
                step_truth, 1, 1, 1
            )  # split_size, b_visual, seq, hidden
            visual_output_r = visual_output_r.view(
                -1, s_visual, h_visual
            )  # split_size*bsz*num_clips, seq, hidden
            video_mask_r = video_mask.unsqueeze(0).repeat(
                step_truth, 1, 1
            )  # split_size, bsz*num_clips, seq
            video_mask_r = video_mask_r.view(
                -1, s_visual
            )  # split_size*bsz*num_clips, seq

            _, _, pooled_output = self.module.get_cross_output(
                sequence_output_l, visual_output_r, attention_mask_l, video_mask_r, 1
            )
            logits = self.similarity_dense(self.dropout(pooled_output))
            retrieve_logits_row = logits.squeeze(-1).view(step_truth, -1)
            retrieve_logits_list.append(retrieve_logits_row)
        retrieve_logits = torch.cat(
            retrieve_logits_list, dim=0
        )  # bsz_text, bsz_video*num_clips
        retrieve_logits = retrieve_logits.view(retrieve_logits.size(0), -1)
        return retrieve_logits  # bsz_text, bsz_video, num_clips

    def _cross_similarity_hard_mining(self, vis_input, cap_input, l1_simi_matrix):
        retrieve_logits = []
        # 聚合所有GPU上的结果
        (sequence_output, attention_mask, text_embed_l1, bsz, caption_input) = cap_input
        (visual_output, video_mask, video_embed_l1, num_clips, img_input) = vis_input
        visual_output = gather_tensor(
            visual_output, method="cat", back_gradient=True, pad_tensors=True
        )
        video_mask = gather_tensor(
            video_mask, method="cat", back_gradient=True, pad_tensors=True
        )  # gather vis
        # 采样topk值
        all_bsz = all_gather(bsz)
        gpu_nid = get_rank()
        beg_idx, end_idx = sum(all_bsz[:gpu_nid]), sum(all_bsz[: gpu_nid + 1])

        for i in range(bsz):
            sequence_output_row = sequence_output[i, :, :].unsqueeze(0)  # seq, hidden
            attention_mask_row = attention_mask[i, :].unsqueeze(0)  # seq
            sequence_output_l = sequence_output_row.repeat(
                bsz, 1, 1
            )  # select, seq, hidden
            attention_mask_l = attention_mask_row.repeat(bsz, 1)  # select, seq
            raw_idx = beg_idx + i  # 恢复原有的idx值
            # 行选择：选择batch中相似度最大的视频【依据横轴的值】
            l1_score_row = l1_simi_matrix[raw_idx]
            if self.config.re_sample_method == "top_k":  # 取L1阶段最高的值作为负例
                l1_score_row[raw_idx] -= 100.0  # 防止负例取到真值
                _, l1_chosen_row = torch.topk(l1_score_row, bsz, sorted=False)
            elif self.config.re_sample_method == "nearliest":  # 取L1阶段与GT（正例）最接近的值为负例
                diag_l1_num = l1_score_row[raw_idx]
                l1_score_row = abs(l1_score_row - diag_l1_num)
                l1_score_row[raw_idx] = 100.0
                _, l1_chosen_row = torch.topk(
                    l1_score_row, bsz, sorted=False, largest=False
                )
            else:
                exit("wrong method!")

            visual_output_r = visual_output[l1_chosen_row]
            visual_output_r[i, :, :] = visual_output[raw_idx, :, :]  # 确保对角线为真值
            video_mask_r = video_mask[l1_chosen_row]
            video_mask_r[i, :] = video_mask[raw_idx, :]
            # forward
            _, _, pooled_output = self.module.get_cross_output(
                sequence_output_l, visual_output_r, attention_mask_l, video_mask_r, 1
            )
            logits = self.similarity_dense(self.dropout(pooled_output)).view(1, -1)
            retrieve_logits.append(logits)

        retrieve_logits = torch.cat(
            retrieve_logits, dim=0
        )  # bsz_text, bsz_video*num_clips
        return retrieve_logits  # bsz_text, bsz_video, num_clips

    def get_mil_nce_loss(self, sim_matrix, batch_size, n_pair=1, weight_vector=None):
        """
        :param sim_matrix: #text(bsz) x #video(bsz*n_pair)
        :return:
        """
        device = sim_matrix.device
        torch_version = get_package_version("torch")
        minor_version = int(torch_version.split(".")[1])
        if minor_version >= 9:
            # need torch1.9 support
            mm_mask = torch.eye(batch_size, device=device)  # bsz * bsz
            mm_mask = torch.kron(
                mm_mask, torch.ones(n_pair, n_pair, device=device)
            ).float()
        else:
            mm_mask = np.eye(batch_size)
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
        mm_mask = torch.tensor(mm_mask, device=device).float()

        # 把mm_mask为1的位置全部设置0， 保留负样本相似度
        from_text_matrix = sim_matrix + mm_mask * -1e12  # n_text, n_video
        from_video_matrix = sim_matrix.transpose(1, 0)  # n_video, n_text

        # from_text_matrix全部为负样本
        new_sim_matrix = torch.cat(
            [from_video_matrix, from_text_matrix], dim=-1
        )  # n_video, #(n_text, n_video)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat(
            [mm_mask, torch.zeros_like(mm_mask)], dim=-1
        )  # 增加负样本的mask
        # 将负样本的概率置0，只保留正样本的概率
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        # mil-nce
        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        # logpt_choice = torch.zeros_like(new_logpt, device=sim_matrix.device)
        mark_ind = torch.arange(batch_size, device=sim_matrix.device) * n_pair + (
            n_pair // 2
        )
        # logpt_choice[mark_ind] = torch.ones(1,device=sim_matrix.device)
        # loss计算重新赋权
        if weight_vector is None:
            sim_loss = new_logpt.index_select(0, mark_ind).mean()
        else:
            sim_loss = torch.mul(
                new_logpt.index_select(0, mark_ind), weight_vector
            ).mean()
        # sim_loss = new_logpt.masked_select(logpt_choice.to(torch.bool)).mean()
        return sim_loss

    def get_l1_simi_matrix(self, text_embed_l1, video_embed_l1, num_clips, cal_cross):
        """
        :param text_embed_l1:  bsz_text, hidden
        :param video_embed_l1: bsz_video * n_clips, hidden
        :return:
              bsz_text, bsz_video, num_clips: clip-level score for training
              bsz_text, num_clips: clip-level score for val/test
        """
        # video-level匹配程度: clip-level匹配程度的聚合: b_t, b_v
        simi_matrix = torch.matmul(
            video_embed_l1.view(-1, num_clips, video_embed_l1.size(-1)),
            text_embed_l1.t(),
        ).permute(
            2, 0, 1
        )  # bsz_text, bsz_video, num_clips
        if not cal_cross:  # vis_input and cap_input must be paired
            # 取对角线输出
            diag_mask = (
                torch.eye(*simi_matrix.shape[:2])
                .unsqueeze(-1)
                .repeat([1, 1, num_clips])
                .to(simi_matrix.device)
            )
            simi_matrix = simi_matrix.masked_select(diag_mask.bool()).view(
                -1, num_clips
            )  # bsz_text, num_clips
            assert simi_matrix.size(0) == text_embed_l1.size(0)
        return simi_matrix

    def get_l2_simi_matrix(
        self,
        cap_embed,
        cap_mask,
        visual_embed,
        visual_mask,
        num_clips,
        cal_cross=False,
    ):
        cap_mask = torch.clamp(cap_mask, 0, 1)
        if cal_cross:
            # clip-level similarity
            # bsz_text, bsz_video, num_clips=1
            simi_matrix = self._cross_similarity(
                cap_embed, visual_embed, cap_mask, visual_mask, num_clips
            )
        else:  # l2_self_simi
            _, _, pooled_output = self.module.get_cross_output(
                cap_embed, visual_embed, cap_mask, visual_mask, 1
            )
            similarity_logits = self.similarity_dense(self.dropout(pooled_output))
            simi_matrix = similarity_logits.view(-1, 1)  # bsz_text, 1
        return simi_matrix

    def get_simi_logits(self, vis_input, cap_input, level, cal_cross):
        """
        :return:
            bsz_text, bsz_video, num_clips: clip-level score for training
            bsz_text, num_clips: clip-level score for val/test
        """
      #  (cap_embed, cap_mask, text_embed_l1, batch_size, caption_input) = cap_input
        (cap_embed, cap_mask, text_embed_l1, batch_size, twm_cap_mask, caption_input) = cap_input
        (visual_embed, visual_mask, video_embed_l1, num_clips, img_input) = vis_input

        l1_simi_loss = None
        if level == "l1":
            if self.training:
                if self.with_moco:
                    if self.moco_utils is None:
                        self.moco_utils = MocoUtils(
                            self.config,
                            img_encoder=self.module.img_encoder,
                            txt_encoder=self.module.text_encoder,
                        ).to(text_embed_l1.device)
                    # compute key features
                    with torch.no_grad():  # no gradient to keys
                        self.moco_utils.momentum_update_key_encoder()

                        # compute visual features
                        visual_embed_dict = self.module.forward_img_encoder(
                            **img_input, img_encoder=self.moco_utils.img_encoder_k
                        )
                        key_v = visual_embed_dict["clip_feature"]  # bsz*n_clips, hidden

                        # compute text features
                        text_embed_dict = self.module.forward_text_encoder(
                            caption_input["caption_raw_input_ids"],
                            caption_input["caption_input_mask"],
                            txt_encoder=self.moco_utils.txt_encoder_k,
                        )
                        key_t = text_embed_dict["pooled_output"]

                    # 1. 对于q=video_embed_l1, k+=key_t, k-=neg_key_t, 跨模态contrastive learning
                    q_v = video_embed_l1  # bsz_txt * num_clips, hidden
                    k_t_pos = key_t  # bsz_txt, hidden
                    k_t_neg = self.moco_utils.txt_queue.clone().detach()  # hidden, K
                    pos = torch.einsum(
                        "bh, bh -> b", [q_v, k_t_pos.repeat_interleave(num_clips, 0)]
                    ).unsqueeze(
                        -1
                    )  # bsz*n_clips, 1
                    neg = torch.einsum("bh, hk -> bk", [q_v, k_t_neg])  # bsz*n_clips, K
                    l1_simi_loss_v = self.moco_utils.moco_loss(pos, neg)
                    # 2. 对于q=text_embed_l1,k+=key_v, k-=self.queue(neg_key_v), 跨模态做 contrastive learning
                    q_t = text_embed_l1  # bsz_text*n_clips, hidden
                    k_v_pos = key_v  # bsz_video*n_clips, hidden
                    # img_queue: n_clips*hidden, K -> k_neg_embed: hidden, n_clips*K
                    k_v_neg = self.moco_utils.img_queue.clone().detach()
                    pos = torch.einsum(
                        "bh, bh -> b", [q_t.repeat_interleave(num_clips, 0), k_v_pos]
                    ).view(
                        -1, num_clips
                    )  # bsz, n_clips
                    neg = torch.einsum("bh, hk -> bk", [q_t, k_v_neg])  # bsz, K
                    l1_simi_loss_t = self.moco_utils.moco_loss(pos, neg)
                    l1_simi_loss = (l1_simi_loss_t + l1_simi_loss_v) / 2.0
                    self.moco_utils.dequeue_and_enqueue(key_v, key_t)
                if get_world_size() > 1:  # gather outputs for larger batch_size
                    text_embed_l1 = gather_tensor(
                        text_embed_l1,
                        method="cat",
                        back_gradient=True,
                        pad_tensors=True,
                    )
                    video_embed_l1 = gather_tensor(
                        video_embed_l1,
                        method="cat",
                        back_gradient=True,
                        pad_tensors=True,
                    )

            simi_matrix = self.get_l1_simi_matrix(
                text_embed_l1, video_embed_l1, num_clips, cal_cross
            )
            return simi_matrix, l1_simi_loss

        elif level == "l2":
            simi_matrix = self.get_l2_simi_matrix(
                cap_embed,
                cap_mask,
                visual_embed,
                visual_mask,
                num_clips,
                cal_cross=cal_cross,
            )
            return simi_matrix
        else:
            raise NotImplementedError

    def reduce_clips(self, simi_logits, level):
        if level == "l1":
            """
            bsz_text, bsz_video, num_clips: clip-level score for training
            bsz_text, num_clips: clip-level score for val/test
            """
            video_logits = simi_logits.logsumexp(-1)
        elif level == "l2":
            # no need to further fuse
            video_logits = simi_logits
        return video_logits

    def forward_stage1(self, vis_input, cap_input, output_dict=None, cal_cross=True):
        output_dict = dict(losses={}) if output_dict is None else output_dict
        num_clips = vis_input[-2]

        # task1: level1 retrieval
        """
        bsz_text, bsz_video, num_clips: clip-level score for training
        bsz_text, num_clips: clip-level score for val/test
        """
        l1_simi, l1_simi_loss = self.get_simi_logits(
            vis_input, cap_input, "l1", cal_cross
        )
        if l1_simi_loss is None:
            if cal_cross and l1_simi.size(0) == l1_simi.size(1):
                # Note:
                # 1. 训练时cal_cross=True, 输入一定是匹配的pair
                # reshape bsz_text, bsz_video, num_clips -> bsz_text*num_clips, bsz_video*num_clips
                bsz_text = bsz_video = l1_simi.size(0)
                mil_simi = (
                    l1_simi.unsqueeze(1)
                    .repeat([1, num_clips, 1, 1])
                    .view(bsz_text * num_clips, bsz_video * num_clips)
                )
                l1_simi_loss = self.get_mil_nce_loss(mil_simi, bsz_text, num_clips)
            else:
                # 2. 测试时计算global simi-matrix分块计算，输入不是匹配的pair, 无需计算loss
                l1_simi_loss = l1_simi.new_tensor(0.0)

        output_dict["losses"]["level1_similarity_loss"] = l1_simi_loss
        output_dict["l1_simi"] = self.reduce_clips(l1_simi, "l1")
        return output_dict

    def forward_stage2(self, vis_input, cap_input, output_dict=None, cal_cross=True):
        output_dict = dict(losses={}) if output_dict is None else output_dict
        batch_size = cap_input[-3]
        # task2: cross-modal retrival
        # clip-level mil-nce loss
        if self.training and self.config.get("hard_example_mining", False):
            # 难例挖掘：列取值
            l1_simi_clone = (
                output_dict["l1_simi"].clone().detach()
            )  # 不需要传递L1 simi matrix的梯度
            l2_simi = self._cross_similarity_hard_mining(
                vis_input, cap_input, l1_simi_clone
            )
        else:
            l2_simi = self.get_simi_logits(vis_input, cap_input, "l2", cal_cross)
        if cal_cross and l2_simi.size(0) == l2_simi.size(1):
            # Note:
            # 1. 训练时输入一定是匹配的pair, clips are already fused in stage2
            # reshape bsz_text, bsz_video, num_clips -> bsz_text*num_clips, bsz_video*num_clips
            mil_simi = (
                l2_simi.unsqueeze(1).repeat([1, 1, 1, 1]).view(batch_size, batch_size)
            )
            if (
                self.training
                and self.config.get("hard_example_mining", False)
                and self.config.re_weight_method == "median"
            ):
                # 难例挖掘：行赋权
                gpu_nid = get_rank()
                all_bsz = all_gather(batch_size)
                beg_idx, end_idx = sum(all_bsz[:gpu_nid]), sum(all_bsz[: gpu_nid + 1])
                l1_diag = torch.diag(l1_simi_clone[beg_idx:end_idx, beg_idx:end_idx])
                l1_median, l1_minimum = torch.mean(l1_diag), torch.min(l1_diag)
                weight_vector = torch.ones(
                    l1_diag.shape, dtype=l1_diag.dtype, device=l1_diag.device
                )
                for i in range(len(l1_diag)):
                    if l1_diag[i] > l1_median:
                        weight_vector[i] = max(
                            (l1_median - l1_minimum) / (l1_diag[i] - l1_minimum), 0.2
                        )
                l2_simi_loss = self.get_mil_nce_loss(
                    mil_simi, batch_size, weight_vector=weight_vector
                )
            else:
                l2_simi_loss = self.get_mil_nce_loss(mil_simi, batch_size)
        else:
            # 1. 测试时计算global simi-matrix分块计算，输入不是匹配的pair, 无需计算loss
            # 2. inference时cal_cross=False, 只需计算pair匹配score,无需计算loss
            l2_simi_loss = l2_simi.new_tensor(0.0)

        output_dict["losses"]["level2_similarity_loss"] = l2_simi_loss
        output_dict["l2_simi"] = self.reduce_clips(l2_simi, "l2")

        return output_dict

    def forward_stage3(self, vis_input, cap_input, output_dict=None, cal_cross=True):
        output_dict = dict(losses={}) if output_dict is None else output_dict
        batch_size = cap_input[-3]
        # task3: cross-modal retrival dmae
        l3_simi, margin_loss = self.dmae_utils.get_similarity_logits(vis_input, cap_input, shaped=True, loose_type=True)
        if cal_cross and l3_simi.size(0) == l3_simi.size(1):
            # Note:
            # 1. 训练时输入一定是匹配的pair, clips are already fused in stage2
            sim_loss1 = self.loss_fct(l3_simi)
            # video2text
            sim_loss2 = self.loss_fct(l3_simi.T)
            l3_simi_loss = (sim_loss1 + sim_loss2) / 2
        else:
            # 1. 测试时计算global simi-matrix分块计算，输入不是匹配的pair, 无需计算loss
            # 2. inference时cal_cross=False, 只需计算pair匹配score,无需计算loss
            l3_simi_loss = l3_simi.new_tensor(0.0)
        output_dict["losses"]["level3_similarity_loss"] = output_dict["losses"]["level3_similarity_loss"] = l3_simi_loss + margin_loss
        output_dict["l3_simi"] = self.reduce_clips(l3_simi, "l2")
        return output_dict

    def forward_stage(self, cap_input, vis_input, cal_cross=True):
        output_dict = None
        if "stage1" in self.config.training_stage:
            output_dict = self.forward_stage1(
                vis_input, cap_input, output_dict, cal_cross=cal_cross
            )
        if "stage2" in self.config.training_stage:
            output_dict = self.forward_stage2(
                vis_input, cap_input, output_dict, cal_cross=cal_cross
            )
        if "stage3" in self.config.training_stage:
            output_dict = self.forward_stage3(vis_input, cap_input, output_dict, cal_cross=cal_cross)
        return output_dict

    def forward(
        self,
        img_input,
        caption_input,
        ocr_input=None,
        region_input=None,
        caption_output=None,
        sample_list=None,
    ):
        if (
            sample_list is not None
            and "text_stage1_output" in sample_list
            and "visual_stage1_output" in sample_list
        ):
            cap_input = sample_list["text_stage1_output"]
            vis_input = sample_list["visual_stage1_output"]
        else:
            cap_input, vis_input, _, _ = self.module.get_l2_input(
                img_input, caption_input
            )
        cap_input = cap_input + (caption_input,)
        vis_input = vis_input + (img_input,)
        # set cal_cross to False for online inference
        return self.forward_stage(cap_input, vis_input, True)

    def get_optimizer_parameters(self, config):
        lr = config.optimizer_attributes.params.lr
        weight_decay = config.optimizer_attributes.params.weight_decay
        encoder_lr_decay = getattr(self.config, "encoder_lr_decay", 0.01)

        # 1e-7 for clip params, 1e-4 for new modules
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        decay_param_tp = [
            (n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ]
        no_decay_param_tp = [
            (n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ]

        def is_pretrain_params(n):
            pretrain_prefix = [
                "text_encoder.embeddings.",
                "text_encoder.encoder.",
                "text_encoder.pooler.",
                "img_embeddings.",
                "img_encoder."
                # "cross_embeddings.",
                # "cross_encoder.",
                # "cross_pooler.",
            ]
            return any([prefix in n for prefix in pretrain_prefix])

        decay_clip_param_tp = [
            (n, p) for n, p in decay_param_tp if is_pretrain_params(n)
        ]
        decay_noclip_param_tp = [
            (n, p) for n, p in decay_param_tp if not is_pretrain_params(n)
        ]

        no_decay_clip_param_tp = [
            (n, p) for n, p in no_decay_param_tp if is_pretrain_params(n)
        ]
        no_decay_noclip_param_tp = [
            (n, p) for n, p in no_decay_param_tp if not is_pretrain_params(n)
        ]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in decay_clip_param_tp],
                "weight_decay": weight_decay,
                "lr": lr * encoder_lr_decay,
            },
            {
                "params": [p for n, p in decay_noclip_param_tp],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in no_decay_clip_param_tp],
                "weight_decay": 0.0,
                "lr": lr * encoder_lr_decay,
            },
            {"params": [p for n, p in no_decay_noclip_param_tp], "weight_decay": 0.0},
        ]
        return optimizer_grouped_parameters
