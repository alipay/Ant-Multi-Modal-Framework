# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import torch.nn.functional as F
from torch import nn
from antmmf.common.registry import registry


def cos_amsoftmax_loss(y_pred, y_true, margin, scale=30.0):
    assert y_true.shape[0] == y_pred.shape[0]
    one_hot = torch.zeros(
        (y_pred.shape[0], y_pred.shape[1]), device=y_pred.device
    ).scatter_(1, y_true.reshape(-1, 1), 1)
    y_am_pred = one_hot * (y_pred - margin) + (1 - one_hot) * y_pred
    y_am_pred *= scale
    return F.cross_entropy(y_am_pred, y_true)


@registry.register_loss("cos_amssoftmax")
class CosAmsSoftmaxLoss(nn.Module):
    """
    cosine amssoftmax loss
    losses:
      - type: cos_amssoftmax
        weight: 1.0
        params:
          margin_params:
            margin_mode: constant
            margin_value: 0.3
    or
    losses:
      - type: cos_amssoftmax
        weight: 1.0
        params:
          margin_params:
            margin_mode: truncate_linear
            margin_start: 0.
            margin_inc: 0.7
            margin_max: 0.25

    """

    def __init__(self, **params):
        super().__init__()
        # support model with multiple CrossEntropyLoss
        self.margin_params = params.pop("margin_params")
        self.margin_mode = self.margin_params.margin_mode
        assert self.margin_mode in ["constant", "truncate_linear"]
        if self.margin_mode == "constant":
            self.margin = self.margin_params.margin_value
        elif self.margin_mode == "truncate_linear":
            self.margin_start = self.margin_params.margin_start
            self.margin_inc = self.margin_params.margin_inc
            self.margin_max = self.margin_params.margin_max
        else:
            raise Exception(
                f"Unkown margin_mode {self.margin_mode} for loss:cos_amssoftmax "
            )

    def forward(self, sample_list, model_output, *args, **kwargs):
        assert "targets" in sample_list
        targets = sample_list["targets"]
        assert "logits" in model_output
        logits = model_output["logits"]
        if self.margin_mode == "constant":
            margin = self.margin
        elif self.margin_mode == "truncate_linear":
            current_epoch = registry.get("current_epoch", 0)
            margin = min(
                self.margin_start + self.margin_inc * current_epoch, self.margin_max
            )

        loss = cos_amsoftmax_loss(logits, targets, margin)
        return loss
