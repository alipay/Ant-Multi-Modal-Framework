# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("nce")
class NCELoss(nn.Module):
    """
    Noise Constrastive Estimation Loss

    The loss is computed in SampledSoftmaxLoss, and is
    used inside a model as a scorer.
    The scorer is used as a last layer to output "sampled" softmax.
    The sampling is done on the output label indices.
    Because of this, computational cost for normalization is reduced.
    In evaluation mode, all of label indices are used to have exact softmax.

    An example of using NCE loss is in MMFusion model.
    Notice that it has a scorer that computes NCE loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output, *args, **kwargs):
        assert "nce_loss" in model_output
        loss = model_output["nce_loss"]
        return loss
