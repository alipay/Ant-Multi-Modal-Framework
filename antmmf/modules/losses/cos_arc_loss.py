# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import numpy as np
import torch
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("cosarc")
class CosArcLoss(nn.Module):
    """
    CosArc: combine CosFace and ArcFace loss
    CosFace: https://arxiv.org/pdf/1801.09414.pdf
    ArcFace: https://arxiv.org/pdf/1801.07698.pdf
    @param cos_margin: cos parameter for CosFace loss
    @param arc_margin: arc parameter for ArcFace loss
    @param scale: scale factor
    """

    def __init__(self, cos_margin, arc_margin, scale=30.0):
        super().__init__()
        self.cos_margin = cos_margin
        self.arc_margin = arc_margin
        self.scale = scale

    def forward(self, sample_list, model_output, eps=1e-12, *args, **kwargs):
        preds = model_output["logits"]
        labels = sample_list.targets
        theta = torch.acos(
            torch.clamp(
                torch.diagonal(preds.transpose(0, 1)[labels]), -1.0 + eps, 1.0 - eps
            )
        )
        theta = torch.clamp(theta, eps, 4.0 * np.arctan(1) - eps)
        numerator = self.scale * (torch.cos(theta + self.arc_margin) - self.cos_margin)
        others = torch.cat(
            [
                torch.cat((preds[idx, :y], preds[idx, y + 1 :])).unsqueeze(0)
                for idx, y in enumerate(labels)
            ],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(
            torch.exp(self.scale * others), dim=1
        )

        return -torch.mean(numerator - torch.log(denominator))
