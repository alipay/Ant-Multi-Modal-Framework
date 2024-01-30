# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("cross_entropy")
class CrossEntropyLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        if params is None:
            params = {}
        # support model with multiple CrossEntropyLoss
        self._logit_key = params.pop("logits_key", "logits")
        self.loss_fn = nn.CrossEntropyLoss(**params)

    def forward(self, sample_list, model_output, *args, **kwargs):
        loss = None
        if "targets" in sample_list:
            loss = self.loss_fn(model_output[self._logit_key], sample_list.targets)
        return loss
