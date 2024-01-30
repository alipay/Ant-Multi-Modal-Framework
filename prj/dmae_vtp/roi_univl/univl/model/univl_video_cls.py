# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from torch import nn

from .univl_video_base import UnivlVideoBase


class UnivlForVideoClassification(nn.Module):
    """
    model that performs classification task
    """

    def __init__(self, config):
        super().__init__()
        self.module = UnivlVideoBase(config, with_cross_encoder=True)
        # for compatibility
        hidden_size = config.get("hidden_size")
        self.clf = nn.Linear(hidden_size, config.num_labels)

        # self.clf = nn.Sequential(
        #     nn.Linear(hidden_size,
        #               hidden_size * 2),
        #     nn.ReLU(True),
        #     nn.Linear(hidden_size * 2, config.num_labels)
        # )

    def forward(
        self,
        img_input,
        caption_input,
        ocr_input=None,
        region_input=None,
        caption_output=None,
        sample_list=None,
    ):
        visual_embed_dict = self.module.forward_img_encoder(**img_input)
        cap_embed, cap_mask, batch_size = self.module.prepare_cross_text(
            caption_input["caption_input_ids"], caption_input["caption_input_mask"]
        )
        visual_embed, visual_mask, num_clip = self.module.prepare_cross_visual(
            visual_embed_dict["visual_embed"], visual_embed_dict["visual_mask"]
        )
        cap_seq_output, visual_seq_output, pooled_output = self.module.get_cross_output(
            cap_embed, visual_embed, cap_mask, visual_mask, 1
        )
        logit = self.clf(pooled_output)
        return {"logits": logit, "out_feat": pooled_output}

    def get_optimizer_parameters(self, config):
        lr = config.optimizer_attributes.params.lr
        weight_decay = config.optimizer_attributes.params.weight_decay

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
                "img_encoder.",
                "cross_embeddings.",
                "cross_encoder.",
                "cross_pooler.",
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
                "lr": lr,
            },
            {
                "params": [p for n, p in decay_noclip_param_tp],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in no_decay_clip_param_tp],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {"params": [p for n, p in no_decay_noclip_param_tp], "weight_decay": 0.0},
        ]
        return optimizer_grouped_parameters
