# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

# pretraining heads with bounded losses & metrics:
# https://github.com/facebookresearch/mmf/blob/master/mmf/models/transformers/heads/itm.py

import torch
from transformers.models.bert.modeling_bert import BertOnlyNSPHead, BertPooler

from antmmf.modules.transformers.heads.base import PredictableHead
from antmmf.common import Configuration, configurable


class ITM(PredictableHead):
    @configurable
    def __init__(
        self,
        hidden_size: int = 768,
        ignore_index: int = -1,
        loss_name: str = "itm_loss",
        with_pooler: bool = True,
    ):
        super().__init__()

        config = Configuration(locals())

        if with_pooler:
            self.pooler = BertPooler(config)
        self.with_pooler = with_pooler
        self.loss_name = loss_name
        self.cls = BertOnlyNSPHead(config)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward_head(
        self, pooled_output=None, encoder_output=None, decoder_output=None
    ):
        """
        :param encoder_output: bsz, seq_length, hidden
        :param decoder_output:
        :return:
        """
        if pooled_output is None:
            assert encoder_output is not None and self.with_pooler
            pooled_output = self.pooler(encoder_output)
        seq_relationship_score = self.cls(pooled_output)
        return seq_relationship_score

    def get_loss_metric(self, predictions, targets):
        """
        TODO: add document here.
        """
        output_dict = {}

        itm_logits = predictions.contiguous().view(-1, 2)
        itm_targets = targets.contiguous().view(-1)

        itm_loss = self.ce_loss(
            itm_logits,
            itm_targets,
        )
        output_dict["losses"] = {}
        output_dict["losses"][self.loss_name] = itm_loss

        with torch.no_grad():
            output_dict["metrics"] = {}
            itm_res = itm_logits.argmax(-1) == itm_targets
            itm_acc = itm_res.sum() / (itm_res.size(0) + 1e-6)
            output_dict["metrics"]["itm_acc"] = itm_acc

        return output_dict
