# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch.nn.functional as F
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("label_smooth_ce")
class LabelSmoothingCrossEntropy(nn.Module):
    """
    label smoothing cross entropy loss
    losses:
     - type: label_smooth_ce
       params:
        eps: 0.1
        reduction: mean
    """

    def __init__(self, **params):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = params.get("eps", 0.1)
        self.reduction = params.get("reduction", "mean")
        self.ignore_index = params.get("ignore_index", -100)
        self.weight = params.get("weight", None)

    def label_smooth_cross_entropy(self, input, target):
        c = input.size()[-1]  # num classes
        log_preds = F.log_softmax(input, dim=-1)
        if self.reduction == "sum":
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == "mean":
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            log_preds,
            target,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
            weight=self.weight,
        )

    def forward(self, sample_list, model_output, *args, **kwargs):
        assert "targets" in sample_list
        targets = sample_list["targets"]
        assert "logits" in model_output
        logits = model_output["logits"]
        loss = self.label_smooth_cross_entropy(logits, targets)
        return loss
