# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch.nn.functional as F
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("kn_dis_loss")
class KnowledgeDistillLoss(nn.Module):
    """
    This loss comes from paper "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, **params):
        super().__init__()
        self._logit_key = params.pop("logits_key", "logits")
        self.T = params.pop("temperature", 1.0)
        self.alpha = params.pop("alpha", 0.7)

    def forward(self, sample_list, model_output, *args, **kwargs):
        student_logits = model_output[self._logit_key]
        teacher_logits = kwargs["teacher_output"][self._logit_key]

        soft_loss = (
            (-F.softmax(teacher_logits, dim=1) * F.log_softmax(student_logits, dim=1))
            .sum(dim=1)
            .mean()
        )

        hard_loss = None
        if "targets" in sample_list:
            hard_loss = F.cross_entropy(student_logits, sample_list.targets)

        return (
            self.alpha * soft_loss + (1 - self.alpha) * hard_loss
            if hard_loss is not None
            else soft_loss
        )
