# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch.nn.functional as F
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("nll_loss")
class NLLLoss(nn.Module):
    """Negative log likelikehood loss."""

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Calculates and returns the negative log likelihood.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `log_prob` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["log_prob"]
        scores = scores.view(-1, scores.size()[-1])

        idx = sample_list["targets"]
        if len(idx.size()) == 2:
            _, idx = idx.max(dim=-1)

        assert scores.size()[0] == len(
            idx
        ), "expected scores in [N, C] and targets in [N]"
        loss = F.nll_loss(scores, idx, reduction="mean")

        return loss
