# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import torch.nn.functional as F
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("wrong")
class WrongLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output, *args, **kwargs):
        pred_score = model_output["logits"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = F.kl_div(res, tar, reduction="mean")
        loss *= target_score.size(1)
        return loss
