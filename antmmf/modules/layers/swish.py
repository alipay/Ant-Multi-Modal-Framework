# -- coding: utf-8 --
# Copyright (c) Ant Group and its affiliates.

import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
