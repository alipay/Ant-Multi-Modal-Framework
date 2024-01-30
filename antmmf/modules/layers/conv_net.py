# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch.nn.functional as F
from torch import nn


class ConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding_size="same",
        pool_stride=2,
        batch_norm=True,
    ):
        super().__init__()

        if padding_size == "same":
            padding_size = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding_size
        )
        self.max_pool2d = nn.MaxPool2d(pool_stride, stride=pool_stride)
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.batch_norm_2d = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.max_pool2d(F.leaky_relu(self.conv(x)))

        if self.batch_norm:
            x = self.batch_norm_2d(x)

        return x
