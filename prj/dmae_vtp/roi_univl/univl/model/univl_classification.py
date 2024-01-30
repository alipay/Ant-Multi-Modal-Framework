# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.modules.encoders.multimodal_bert_clf import MultimodalBertClf
from .univl_base import UniVlBase


class UnivlBertForClassification(MultimodalBertClf):
    """
    model that performs classification task
    """

    def __init__(self, config):
        super().__init__(config, modal_encoder=UniVlBase)

    def forward(
        self,
        img_input,
        caption_input,
        ocr_input=None,
        region_input=None,
        caption_output=None,
        sample_list=None,
    ):
        model_output = self.enc(
            img_input,
            caption_input,
            ocr_input,
            region_input,
            None,
            sample_list=None,
        )
        cap_enc_embed, ocr_enc_embed, visual_enc_embed, text_enc_embed = model_output[
            "enc_output"
        ]
        # dec_output = model_output["dec_output"]
        # head_output = model_output["head_output"]

        pooled_output = cap_enc_embed[:, 0]
        return {"logits": self.clf(pooled_output)}
