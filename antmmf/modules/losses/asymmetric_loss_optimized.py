# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("asymmetric_loss")
class AsymmetricLossOptimized(nn.Module):
    """
    refer to paper: Asymmetric Loss For Multi-Label Classification
    https://arxiv.org/abs/2009.14119
    """

    def __init__(self, **params):
        super().__init__()
        if params is None:
            params = {}
        # support model with multiple CrossEntropyLoss
        self._logit_key = params.pop("logits_key", "logits")

        self.gamma_neg = params.pop("gamma_neg", 4)
        self.gamma_pos = params.pop("gamma_pos", 1)
        self.clip = params.pop("clip", 0.05)
        self.disable_torch_grad_focal_loss = params.pop(
            "disable_torch_grad_focal_loss", False
        )
        self.eps = 1e-8
        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = (
            self.anti_targets
        ) = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, sample_list, model_output):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        x = model_output[self._logit_key]
        y = sample_list["targets"]

        return self.asy_loss(x, y)

    def asy_loss(self, x, y):
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos
        # print(self.xs_pos,self.xs_neg)
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
