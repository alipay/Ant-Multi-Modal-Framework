# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn


class GatedTanh(nn.Module):
    r"""
    From: https://arxiv.org/pdf/1707.07998.pdf
    nonlinear_layer (f_a) : x\in R^m => y \in R^n # noqa
    \tilda{y} = tanh(Wx + b)
    g = sigmoid(W'x + b')
    y = \tilda(y) \circ g # noqa
    input: (N, *, in_dim)
    output: (N, *, out_dim)
    """

    def __init__(self, in_dim, out_dim):
        super(GatedTanh, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gate_fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y_tilda = torch.tanh(self.fc(x))
        gated = torch.sigmoid(self.gate_fc(x))

        # Element wise multiplication
        y = y_tilda * gated

        return y
