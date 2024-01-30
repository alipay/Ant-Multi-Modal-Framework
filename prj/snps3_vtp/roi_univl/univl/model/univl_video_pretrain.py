# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import copy
import random

import numpy as np
import torch
import torch as T
import torch.nn.functional as F
from torch import nn

from antmmf.modules.build import build_transformer_head
from antmmf.modules.encoders import TextEncoder
from .univl_video_ret import UnivlForVideoTextRetrieval


class VideoMLMPretrainHeads(nn.Module):
    """
    Pretraining head for univl model,
    1. Masked language(ocr and caption) modelling
    2. Frame Order Modelling
    """

    def __init__(self, config):
        super().__init__()

        self.transformer_mlm = build_transformer_head(
            config.pretraining_heads.transformer_mlm
        )
        self.text_encoder_mlm = build_transformer_head(
            config.pretraining_heads.text_encoder_mlm
        )

    def forward(
        self,
        text_enc_embed,  # text encoder output
        text_cross_embed,
        text_lm_label_ids,
        num_clips,
    ):
        output = []
        # task1: Hierarchical Masked Language Modelling for caption
        # task1-Hier1: text encoder mlm for caption & ocr
        text_encoder_mlm_output = self.text_encoder_mlm(
            encoder_output=text_enc_embed,
            targets=text_lm_label_ids,
        )
        output.append(text_encoder_mlm_output)
        # task1-Hier2: multi-modal transformer encoder mlm for caption
        if text_cross_embed is not None:
            transformer_mlm_output = self.transformer_mlm(
                encoder_output=text_cross_embed,
                targets=text_lm_label_ids,
            )
            output.append(transformer_mlm_output)

        return output


class UnivlForVideo(UnivlForVideoTextRetrieval):
    def __init__(self, config):
        super().__init__(config)
        if self.config.with_temporal_encoder:
            self.add_temporal_head()

    def add_temporal_head(self):
        # refer to STN implementation:
        # https://github.com/bomri/SlowFast/blob/master/slowfast/models/video_model_builder.py#L801
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.hidden_size))

        self.temporal_encoder = TextEncoder(self.config.temporal_encoder).module
        self.temporal_encoder.embeddings.word_embeddings = None
        self.temporal_encoder.pooler = None

    def get_temporal_output(self, clip_feat):
        # B, num_clips, hidden
        bsz, n_clips, hidden = clip_feat.shape
        attention_mask = torch.ones(
            (bsz, n_clips), dtype=torch.long, device=clip_feat.device
        )
        cls_atten = torch.ones(1).expand(bsz, -1).to(clip_feat.device)
        attention_mask = torch.cat((cls_atten, attention_mask), dim=1)

        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        input_embeds = torch.cat((cls_tokens, clip_feat), dim=1)
        seq_out, _ = self.temporal_encoder(
            inputs_embeds=input_embeds, attention_mask=attention_mask
        )
        return seq_out

    def forward(
        self,
        img_input,
        caption_input,
        ocr_input=None,
        region_input=None,
        caption_output=None,
        sample_list=None,
    ):
        (
            cap_input,
            vis_input,
            text_embed_dict,
            visual_embed_dict,
        ) = self.module.get_l2_input(img_input, caption_input)
        cap_embed, cap_mask, text_embed_l1, batch_size = cap_input
        visual_embed, visual_mask, video_embed_l1, num_clips = vis_input

        # clip-level output
        cap_seq_output, visual_seq_output, clip_feat = self.module.get_cross_output(
            cap_embed, visual_embed, cap_mask, visual_mask, num_clips
        )
        seq_out = self.get_temporal_output(clip_feat)
        # # cal stage1 & stage2 mil-nce loss
        # mil_nce_output = self.forward_stage(cap_input, vis_input, True)
        return seq_out


class MLPLayer(nn.Module):
    def __init__(self, in_hsz, out_hsz):
        super(MLPLayer, self).__init__()
        self.linear_1 = nn.Linear(in_hsz, in_hsz * 2)
        self.LayerNorm = nn.LayerNorm(in_hsz * 2, eps=1e-5)
        self.linear_2 = nn.Linear(in_hsz * 2, out_hsz)
        self.act = nn.GELU()

    def forward(self, x):
        x_1 = self.linear_1(x)
        x_1 = self.act(x_1)
        x_1 = self.LayerNorm(x_1)
        x_2 = self.linear_2(x_1)
        return x_2


class UnivlForVideoPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = UnivlForVideo(config)
        self.arch_type = self.config.get("arch_type", "univl")

        # MLM head
        self.mlm_head = VideoMLMPretrainHeads(config)
        # tie weights for MLM head
        self.mlm_head.transformer_mlm.tie_weights(
            self.model.module.text_encoder.embeddings.word_embeddings
        )
        self.mlm_head.text_encoder_mlm.tie_weights(
            self.model.module.text_encoder.embeddings.word_embeddings
        )

        # ITM head
        self.itm_head = build_transformer_head(config.pretraining_heads.itm)

        if self.config.with_temporal_encoder and "stage2" in self.config.training_stage:
            # clip order head
            self.fom_output = MLPLayer(config.hidden_size, config.max_clip_len)

    def forward(
        self,
        img_input,
        caption_input,
        ocr_input=None,
        region_input=None,
        caption_output=None,
        sample_list=None,
    ):
        (
            cap_input,
            vis_input,
            text_embed_dict,
            visual_embed_dict,
        ) = self.model.module.get_l2_input(img_input, caption_input)
        cap_embed, cap_mask, text_embed_l1, batch_size = cap_input
        visual_embed, visual_mask, video_embed_l1, num_clips = vis_input

        pretrain_output = []
        # MIL-NCE task: cal stage1 & stage2 mil-nce loss
        cap_input = cap_input + (caption_input,)
        vis_input = vis_input + (img_input,)
        mil_nce_output = self.model_similarity(cap_input, vis_input, caption_input, sample_list)
        pretrain_output += [mil_nce_output]

        # clip-level output
        cap_lm_label_ids = sample_list.caption_lm_label_ids
        if "stage2" in self.config.training_stage:
            cross_input_t = cap_embed
            if (
                self.training
                and np.random.choice([0, 1]) == 0
                and self.arch_type == "univl"
            ):  # attentive masking
                wim = text_embed_dict["words_importance"]
                cap_raw = caption_input["caption_raw_input_ids"].clone().detach()
                cap_lm_label_ids = -1 * T.ones(
                    cap_raw.size(), dtype=T.long, device=cap_raw.device
                )

                # set special token to zero
                spc_txt = T.logical_or(
                    T.logical_or(cap_raw == 101, cap_raw == 102), cap_raw == 0
                )
                wim[T.where(spc_txt)] = 0.0
                bsz = cap_raw.size(0)

                for i in range(bsz):
                    # 有效tokens数
                    num_tokens = spc_txt[i].logical_not().sum()
                    pos = (
                        T.multinomial(wim[i] + 1e-6, max(int(num_tokens * 0.15), 1))
                        .cpu()
                        .numpy()
                    )

                    for p in pos:
                        cap_lm_label_ids[i][p] = cap_raw[i][p]
                        cap_raw[i][p] = 103

                # inplace update
                cross_input_t, _, _ = self.model.module.prepare_cross_text(
                    cap_raw, cap_mask
                )

            cross_visual_embed = visual_embed_dict["visual_embed"]
            cross_input_v, cross_input_m, _ = self.model.module.prepare_cross_visual(
                cross_visual_embed, visual_embed_dict["visual_mask"]
            )
            (
                cross_output_t,
                cross_output_v,
                cross_pooled_output,
            ) = self.model.module.get_cross_output(
                cross_input_t, cross_input_v, cap_mask, cross_input_m, 1
            )
        else:
            cross_output_t = None

        # MLM task: cal mlm loss & metric
        mlm_output = self.mlm_head(
            text_embed_dict["sequence_output"],
            cross_output_t,
            cap_lm_label_ids,
            num_clips,
        )
        pretrain_output += mlm_output

        # =======> add itm loss <=======
        if "stage2" in self.config.training_stage:
            # forward negtaive samples
            cap_false_input_ids = caption_input.get("caption_false_input_ids")
            cap_false_input_mask = caption_input.get("caption_false_input_mask")
            if cap_false_input_ids is not None and cap_false_input_mask is not None:
                (
                    false_cap_embed,
                    false_cap_mask,
                    _,
                ) = self.model.module.prepare_cross_text(
                    cap_false_input_ids, cap_false_input_mask
                )
                _, _, clip_neg_feat = self.model.module.get_cross_output(
                    false_cap_embed,
                    visual_embed,
                    false_cap_mask,
                    visual_mask,
                    1,
                )
                # label 数据打散 与 label生成
                device = clip_neg_feat.device
                num_pos, num_neg = cross_pooled_output.size(0), clip_neg_feat.size(0)
                itm_labels = torch.cat(
                    [torch.ones(num_pos).long(), torch.zeros(num_neg).long()]
                ).to(device)
                itm_feats = torch.cat([cross_pooled_output, clip_neg_feat], 0)
                shuffle_idx = torch.randperm(itm_labels.size(0))
                itm_labels, itm_feats = itm_labels[shuffle_idx], itm_feats[shuffle_idx]
                itm_output = self.itm_head(pooled_output=itm_feats, targets=itm_labels)
                pretrain_output += [itm_output]
        # =======> end itm loss <=======

        if self.config.with_temporal_encoder and "stage2" in self.config.training_stage:
            # COM task: clip order loss & metric
            cross_pooled_output = cross_pooled_output.view(
                -1, num_clips, cross_pooled_output.size(-1)
            )
            fom_output_dict = self.model_clip_order(cross_pooled_output)
            pretrain_output += [fom_output_dict]

        return pretrain_output

    def model_similarity(self, cap_input, vis_input, caption_input=None, sample_list=None):
        mil_nce_output = self.model.forward_stage(cap_input, vis_input, True,
                                                  caption_input=caption_input, sample_list=sample_list)

        def cal_ret_metric(x):
            # TODO:here is cpu copy and cuda cudaStreamSynchronize,replace it with pure torch
            neg_x = -x
            diag_neg_x = torch.diag(neg_x)
            (w,) = diag_neg_x.size()
            # validate using torch.allclose True
            ind = torch.sort(neg_x, axis=1)[0] - diag_neg_x.view(w, 1)
            # ind = np.sort(-x, axis=1) - np.diag(-x)[:, np.newaxis]
            # ind = np.where(ind == 0)[1]
            ind = torch.where(ind == 0)[1]

            def _recall(topk):
                return torch.sum(ind < int(topk)) / (len(ind) + 1e-10)

            mr = torch.median(ind) + 1
            r_1 = _recall(1)
            r_5 = _recall(5)
            r_10 = _recall(10)
            return mr, r_1, r_5, r_10

        with torch.no_grad():
            device = mil_nce_output["l1_simi"].device
            l1_mr, l1_r1, l1_r5, l1_r10 = cal_ret_metric(mil_nce_output["l1_simi"])
            mil_nce_output["metrics"] = {
                "l1_mr": l1_mr.to(device),
                "l1_r_1": l1_r1.to(device),
                "l1_r_5": l1_r5.to(device),
                "l1_r_10": l1_r10.to(device),
            }
            if "l2_simi" in mil_nce_output:
                l2_mr, l2_r1, l2_r5, l2_r10 = cal_ret_metric(mil_nce_output["l2_simi"])
                mil_nce_output["metrics"].update(
                    {
                        "l2_mr": l2_mr.to(device),
                        "l2_r_1": l2_r1.to(device),
                        "l2_r_5": l2_r5.to(device),
                        "l2_r_10": l2_r10.to(device),
                    }
                )
            '''Extra outputs for SNP-S3'''
            if "VWM_after_simi" in mil_nce_output:
                device = mil_nce_output["VWM_after_simi"].device
                la_mr, la_r1, la_r5, la_r10 = cal_ret_metric(mil_nce_output["VWM_after_simi"])
                mil_nce_output["metrics"].update(
                    {
                        "VWM_after_mr": la_mr.to(device),
                        "VWM_after_r@1": la_r1.to(device),
                        "VWM_after_r@5": la_r5.to(device),
                        "VWM_after_r@10": la_r10.to(device),
                    }
                )
            if "VWM_before_simi" in mil_nce_output:
                lb_mr, lb_r1, lb_r5, lb_r10 = cal_ret_metric(mil_nce_output["VWM_before_simi"])
                mil_nce_output["metrics"].update(
                    {
                        "VWM_before_mr": lb_mr.to(device),
                        "VWM_before_r@1": lb_r1.to(device),
                        "VWM_before_r@5": lb_r5.to(device),
                        "VWM_before_r@10": lb_r10.to(device),
                    }
                )
        return mil_nce_output

    def model_clip_order(self, clip_feat):
        # refer to:
        # https://github.com/linjieli222/HERO/blob/master/model/model.py#L306-L336
        # 数据集输入的clip按顺序排列
        bsz, num_clips, hidden = clip_feat.size()
        clip_idx = torch.arange(num_clips)

        # shuffle clip orders
        b_order, b_targets = [], []
        for _ in range(bsz):
            shuf_o, shuf_t = self.random_reorder(clip_idx, 0.15)
            b_order.append(torch.as_tensor(shuf_o).to(clip_feat.device))
            b_targets.append(torch.as_tensor(shuf_t).to(clip_feat.device))
        shuffled_orders, shuffled_targets = torch.stack(b_order, 0), torch.stack(
            b_targets, 0
        ).view(-1)

        shuffled_orders = shuffled_orders.unsqueeze(-1).expand_as(clip_feat)
        clip_feat_new = clip_feat.new_zeros(*clip_feat.size())
        shuffled_clip_feat = clip_feat_new.scatter_(1, shuffled_orders, clip_feat)

        encoded_clip = self.model.get_temporal_output(shuffled_clip_feat)
        clip_rep = encoded_clip[:, 1:]  # 去掉CLS token
        clip_rep = clip_rep.contiguous().view(-1, clip_rep.size(-1))
        frame_reorder_outputs = self.fom_output(clip_rep)
        fom_loss = F.cross_entropy(
            frame_reorder_outputs, shuffled_targets, ignore_index=-1, reduction="mean"
        )
        with torch.no_grad():
            pred_res = (frame_reorder_outputs.argmax(-1) == shuffled_targets)[
                torch.where(shuffled_targets != -1)
            ]
            fom_acc = pred_res.sum() / (pred_res.size(0) + 1e-6)
        fom_output_dict = {
            "losses": {"fom_loss": fom_loss},
            "metrics": {"fom_acc": fom_acc},
        }
        return fom_output_dict

    def random_reorder(self, pos_ids, random_reorder_p=0.15):
        """
        random reorder frame positions
        copy from:
        https://github.com/linjieli222/HERO/blob/master/data/fom.py#L96
        """
        selected_pos = []
        target_pos = []
        for i, pos_id in enumerate(pos_ids):
            prob = random.random()
            # mask token with 15% probability
            if prob < random_reorder_p:
                selected_pos.append(i)
                target_pos.append(pos_id)
        target_pos_shuffled = copy.deepcopy(target_pos)
        random.shuffle(target_pos_shuffled)
        output_order = copy.deepcopy(pos_ids)
        output_target = [-1] * len(output_order)
        for i, pos in enumerate(selected_pos):
            output_order[pos] = target_pos_shuffled[i]
            output_target[target_pos_shuffled[i]] = pos
        return output_order, output_target

    def get_optimizer_parameters(self, config):
        lr = config.optimizer_attributes.params.lr
        weight_decay = config.optimizer_attributes.params.weight_decay
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        decay_param_tp = [
            (n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ]
        no_decay_param_tp = [
            (n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in decay_param_tp],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in no_decay_param_tp],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
        return optimizer_grouped_parameters
