# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from antmmf.common import Configuration, configurable


class MRC(nn.Module):
    """VilBERT style Region-Head that performs region classification task.

    This implementation refers to:
    https://github.com/e-bug/volta/blob/9e5202141920600d58a9c5c17519ca453795d65d/volta/encoders.py#L720
    """

    @configurable
    def __init__(
        self,
        vocab_size: int = 80,  # coco classes
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-6,
        hidden_act: str = "gelu",
        ignore_index: int = -1,
        loss_name: str = "masked_region_classification",
    ):
        super().__init__()

        # Head modules
        self.cls = BertOnlyMLMHead(Configuration(locals()))
        self.vocab_size = vocab_size
        self.loss_name = loss_name

        # Loss
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

    def tie_weights(self, module: Optional[torch.nn.Module] = None):
        self.cls.predictions.decoder.weight = module.weight

    def forward(
        self,
        sequence_output: torch.Tensor,
        region_to_predict: torch.Tensor,
        region_cls_dis: Optional[torch.Tensor] = None,
    ):
        """
         refer to: https://github.com/e-bug/volta/blob/9e5202141920600d58a9c5c17519ca453795d65d/volta/losses.py#L16
        :param sequence_output: b, num_region_length, num_classes
        :param region_to_predict: b, num_region_length
        :param region_cls_dis: b, num_region_length, num_classes
        :return:
        """
        prediction = self.cls(sequence_output)
        loss = self.kl_loss(F.log_softmax(prediction, dim=2), region_cls_dis)
        region_loss = torch.sum(
            loss * (region_to_predict == 1).unsqueeze(2).float()
        ) / max(torch.sum((region_to_predict == 1)), 1)

        output_dict = {"losses": {}, "metrics": {}}
        output_dict["losses"][self.loss_name] = region_loss

        # MRC  acc
        with torch.no_grad():
            mrc_res = (
                (region_cls_dis.argmax(-1) == prediction.argmax(-1)).to(torch.float)
                * (region_to_predict == 1)
            ).sum()

            mrc_acc = mrc_res / max(torch.sum((region_to_predict == 1)), 1)

            output_dict["losses"][self.loss_name] = region_loss
            output_dict["metrics"]["mrc_acc"] = mrc_acc
        return output_dict
