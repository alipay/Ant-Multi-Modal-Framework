# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch.nn.functional as F
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("logit_bce")
class BinaryCrossEntropyWithLogits(nn.Module):
    """Returns Binary Cross Entropy for logits.

    Attention:
        `Key`: logit_bce
    """

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Calculates and returns the binary cross entropy for logits

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["logits"]
        targets = sample_list["targets"]
        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="mean")

        return loss * targets.size(1)
