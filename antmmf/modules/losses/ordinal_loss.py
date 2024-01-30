# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from antmmf.common.registry import registry


@registry.register_loss("ordinal_loss")
class OrdinalLoss(torch.nn.Module):
    """Creates an ordinal criterion that measures the Implicit Constraints
    between each element in the input and ordinal target.
    The loss module contains K-1 ordered thresholds for K ordinal levels,
    The K-1 thresholds are tunable paprameters updated via autograd.
    The Implicit Constraints are designed to guarantee properly ordered thresholds.
    Reference to 'support vector ordinal regression' Chu & Keerthi 2007 for more details.

    Args:
        ordinal (int, required): The number of ordinal levels in the learning task. Default: 3
    Shape:
        - Input: regression network output of any number of dimensions.
        - Target: integers less than K for K zero-based ordinal levels, same shape as the input.
        - Output: scalar.
    Examples:
        >>> loss = OrdinalLoss(ordinal=10)
        >>> output = dict()
        >>> output['logits'] = torch.randn(4,1)
        >>> label = dict()
        >>> label['targets'] = torch.tensor([[0], [1], [2], [3]])
        >>> output = loss(label, output)
        >>> pred = loss.pred(output['logits'])
        see /prj/ordinal/ordinal_example.py for an example of usage
    """

    def __init__(self, ordinal=3):
        super(OrdinalLoss, self).__init__()
        assert ordinal > 1, "ordinal is required to be at least 2."
        self.threshold = torch.nn.Parameter(
            torch.linspace(-ordinal / 2.0, ordinal / 2.0, ordinal - 1)
        )

    def forward(self, sample_list, model_output):
        assert "targets" in sample_list, "sample_list should contain targets."
        target = sample_list["targets"]
        assert "logits" in model_output, "model_output should contain logits."
        input = model_output["logits"]
        # input and target should be of the same size
        assert (
            target.size() == input.size()
        ), "Using a target size that is different to the input size. \
            This will likely lead to incorrect results due to broadcasting."
        epsilon = input.view(-1).unsqueeze(-1) - self.threshold
        mask = torch.zeros(
            epsilon.shape[0], len(self.threshold) + 1, device=self.threshold.device
        )
        mask[(torch.arange(epsilon.shape[0]), target.view(-1))] = 2
        mask = 1 - mask.cumsum(dim=-1)
        # the last column of mask unused in large margin loss computation
        loss = torch.max(1.0 - epsilon * mask[:, :-1], torch.zeros_like(epsilon))
        return loss.mean()

    def __str__(self):
        """Print out the threshold values"""
        return f"ordinal thresholds = {self.threshold}"

    def pred(self, input):
        """Convert input values to ordinal levels via thresholding
        - Input: function values.
        - Output: int, predicted ordinal levels, [0, ordinal).
        """
        return ((input.unsqueeze(-1) - self.threshold) > 0).type(torch.int).sum(-1)
