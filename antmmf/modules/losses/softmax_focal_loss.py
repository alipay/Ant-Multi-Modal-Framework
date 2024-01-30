# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
import torch.nn.functional as F


class SoftmaxFocalLoss(nn.Module):
    """
    Softmax version of Focal Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    The implementation is based on: https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    Args:
        inputs: A float tensor of shape (N, C), where C is the number of classes.
                The predictions for each example.
        targets: A float tensor of shape (N, ). Stores the value of
                 classification label (0, 1, ..., C - 1) for each element in inputs.
        alpha: (optional) Weighting factor in range (0, 1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        ignore_index: Specifies a target value that is ignored and does not contribute
                      to the input gradient.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """

    def __init__(
        self,
        alpha: float = -1,
        gamma: float = 2,
        ignore_index: int = -100,
        reduction: str = "none",
        logits_key: str = "logits",
        targets_key: str = "targets",
    ):
        super(SoftmaxFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.logits_key = logits_key
        self.targets_key = targets_key

    def forward(self, sample_list, model_output, *args, **kwargs):
        assert self.targets_key in sample_list, "sample_list should contain targets."
        targets = sample_list[self.targets_key]  # (N,)
        assert self.logits_key in model_output, "model_output should contain logits."
        inputs = model_output[self.logits_key]  # (N, C)

        # extract valid parts
        valid_inds = (targets != self.ignore_index).nonzero().view(-1)
        targets = targets[valid_inds]
        inputs = inputs[valid_inds]

        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # (N,)
        p = F.softmax(inputs)
        pt = p.gather(1, targets.view(-1, 1)).view(-1)  # (N, C) -> (N)
        loss = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha = torch.tensor(
                [self.alpha, 1 - self.alpha], device=inputs.device, dtype=inputs.dtype
            )
            alpha_t = alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
