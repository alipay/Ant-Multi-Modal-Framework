# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn

from .univl_video_base import UnivlVideoBase


class UnivlForVideoMultiChoiceQA(nn.Module):
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

    def forward(
        self,
        img_input,
        caption_input,
        ocr_input=None,
        region_input=None,
        caption_output=None,
        sample_list=None,
    ):
        cap_input_list, vis_input_list = self.get_l2_input_for_mc_qa(
            img_input, caption_input
        )
        simi_vector_list = []
        for vis_input, cap_input in zip(vis_input_list, cap_input_list):
            (cap_embed, cap_mask) = cap_input
            (visual_embed, visual_mask) = vis_input
            _, _, pooled_output = self.module.get_cross_output(
                cap_embed, visual_embed, cap_mask, visual_mask, 1
            )
            similarity_logits = self.similarity_dense(self.dropout(pooled_output))
            simi_vector = similarity_logits.view(-1, 1).t()  # [1, caption总数]
            simi_vector_list.append(simi_vector)
        simi_vector_all = torch.cat(simi_vector_list, dim=0)
        return {"logits": simi_vector_all}

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

    def get_l2_input_for_mc_qa(self, raw_img_input, raw_caption_input):
        img_key_list = [
            "image_data",
            "image_pad_mask",
            "image_n_clips",
            "image_num_frames",
        ]
        img_input_list = []
        bsz = len(raw_img_input["image_n_clips"])
        # 第一步，将输入的image特征，按照batch进行切分
        for i in range(bsz):
            per_img_dict = {}
            for img_key in img_key_list:
                if img_key in ["image_data", "image_pad_mask"]:
                    assert len(raw_img_input[img_key][i]) == 1
                    feat = raw_img_input[img_key][i][0].unsqueeze(dim=0)
                    per_img_dict[img_key] = feat
                else:
                    per_img_dict[img_key] = raw_img_input[img_key][i]
            img_input_list.append(per_img_dict)

        cap_input_list = raw_caption_input["caption_options"]
        cap_length_list = raw_caption_input["caption_length"]
        cap_output_list = []
        vis_output_list = []

        for img_info, cap_info_r, len_cap in zip(
            img_input_list, cap_input_list, cap_length_list
        ):
            assert len(cap_info_r) == 1
            cap_info = cap_info_r[0]
            len_choice = len_cap
            raw_visual_embed_dict = self.module.forward_img_encoder(**img_info)
            # 复制得到的image-level feature，使之维度与caption choice的数目相同
            new_visual_embed = torch.repeat_interleave(
                raw_visual_embed_dict["visual_embed"], repeats=len_choice, dim=0
            )
            new_visual_mask = torch.repeat_interleave(
                raw_visual_embed_dict["visual_mask"], repeats=len_choice, dim=0
            )

            cap_embed, cap_mask, batch_size = self.module.prepare_cross_text(
                cap_info["caption_input_ids"], cap_info["caption_input_mask"]
            )
            visual_embed, visual_mask, num_clips = self.module.prepare_cross_visual(
                new_visual_embed, new_visual_mask
            )
            cap_input = (cap_embed, cap_mask)
            vis_input = (visual_embed, visual_mask)
            cap_output_list.append(cap_input)
            vis_output_list.append(vis_input)
        # 输出为若干list，每个list代表一个batch的caption，vision特征和对应的标签
        return cap_output_list, vis_output_list
