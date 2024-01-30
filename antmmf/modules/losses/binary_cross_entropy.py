# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("bce")
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = torch.nn.BCELoss()

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Calculates and returns the binary cross entropy.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["logits"]
        targets = sample_list["targets"]
        loss = self.loss(scores, targets)

        return loss
