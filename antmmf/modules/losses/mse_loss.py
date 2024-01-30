# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("mse")
class MSELoss(nn.Module):
    """
    MSE loss
    """

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss(reduce=True, reduction="sum")

    def forward(self, sample_list, model_output):
        assert "targets" in sample_list
        targets = sample_list["targets"]
        assert "logits" in model_output
        logits = model_output["logits"]

        loss = self.loss(targets, logits)
        return loss
