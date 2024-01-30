# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from antmmf.common.registry import registry
from antmmf.modules.metrics.mce_accuracy import MCEAccuracy


@registry.register_metric("asm_metric")
class AsymMetric(MCEAccuracy):
    def __init__(self, name: str = "asm_metric"):
        super(MCEAccuracy, self).__init__(name=name)

    def _get_pred_label(self, logits):
        return torch.where(torch.sigmoid(logits) >= 0.5, 1, 0).long()
