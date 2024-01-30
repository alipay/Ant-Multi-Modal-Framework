# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("pairwise_loss")
class PairwiseLoss(nn.Module):
    """
    Compute pairwise loss.

    Args:
      x: [N, D] float tensor, representations for N examples.
      y: [N, D] float tensor, representations for another N examples.
      labels: [N] int tensor, with values in -1 or +1.  labels[i] = +1 if x[i]
        and y[i] are similar, and -1 otherwise.
      loss_type: margin or hamming.
      margin: float scalar, margin for the margin loss.

    Returns:
      loss: [N] float tensor.  Loss for each pair of representations.

    Docs:
        https://paperswithcode.com/paper/graph-matching-networks-for-learning-the-1#code
        prepare for matching tasks
    """

    def __init__(self, **params):
        super().__init__()
        if params is None:
            params = {}
        self.loss_type = params.get("loss_type", "margin")
        self.margin = params.get("margin", 1.0)

    @staticmethod
    def euclidean_distance(x, y):
        """This is the squared Euclidean distance."""
        return torch.sum((x - y) ** 2, dim=-1)

    @staticmethod
    def approximate_hamming_similarity(x, y):
        """Approximate Hamming similarity."""
        return torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)

    def forward(self, x, y, labels, *args, **kwargs):
        labels = labels.float()
        if self.loss_type == "margin":
            return torch.relu(
                self.margin - labels * (1 - PairwiseLoss.euclidean_distance(x, y))
            )
        elif self.loss_type == "hamming":
            return (
                0.25 * (labels - PairwiseLoss.approximate_hamming_similarity(x, y)) ** 2
            )
        else:
            raise ValueError("Unknown loss_type %s" % self.loss_type)
