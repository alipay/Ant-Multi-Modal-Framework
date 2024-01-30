# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import numpy as np
from torch import nn
from .graph_decoder import GraphDecoder


@GraphDecoder.register()
class DeltaClassifyDecoder(nn.Module):
    """
    Multi fully connect layers for classification problem

    Args:
        input_dim (int): dimension of input tensor, default value is 128.
        num_class (int): tht number of classes, default value is 2.
        layer_num (int): the number of fully connect layers, [512, 256, 2] means two fc layers, default value is 2.
        dropout_ratio (float): ratio of dropout, default value is 0.3.
        short_cut (bool): add short cut path between input and output, default value is False.
        bias (bool): whether use bias or not in Linear.
        use_bn (bool): whether add batch normalization layer or layer normalization between Linear and ReLU layers.

    """

    def __init__(
        self,
        input_dim: int = 128,
        num_class: int = 2,
        layer_num: int = 2,
        dropout_ratio: float = 0.3,
        short_cut: bool = False,
        bias: bool = True,
        use_bn: bool = True,
    ):
        super(DeltaClassifyDecoder, self).__init__()
        channels = list(np.linspace(input_dim, num_class, layer_num + 1, dtype=int))

        blocks = []
        for i in range(layer_num):
            blocks.append(nn.Linear(channels[i], channels[i + 1], bias=bias))
            if use_bn:
                blocks.append(nn.BatchNorm1d(channels[i + 1]))
            else:
                blocks.append(nn.LayerNorm(channels[i + 1]))

            blocks.append(nn.ReLU(inplace=True))
            if dropout_ratio > 0:
                blocks.append(nn.Dropout(p=dropout_ratio))

        # Last layer which output the logits doesnt need LN, Dropout and relu
        blocks = blocks[: (-3 if dropout_ratio > 0 else -2)]

        if layer_num == 0:
            blocks = [torch.nn.Identity()]
        self.blocks = nn.Sequential(*blocks)

        if short_cut and layer_num:
            self.short_cut = nn.Linear(channels[0], channels[-1], bias=bias)

    def forward(self, x):
        out = self.blocks(x)
        if hasattr(self, "short_cut"):
            out = out + self.short_cut(x)
        return out
