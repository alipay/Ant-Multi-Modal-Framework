# -- coding: utf-8 --

import torch
from torch import nn
from .graph_decoder import GraphDecoder


@GraphDecoder.register()
class ClassifyDecoder(nn.Module):
    def __init__(
        self,
        node_embed_dim: int,
        num_classes: int = 2,
        layer_num: int = 1,
    ):
        super(ClassifyDecoder, self).__init__()
        channels = (
            [node_embed_dim] + [node_embed_dim // 4] * (layer_num - 1) + [num_classes]
        )

        layers = []
        for idx in range(layer_num):
            layers.append(nn.Linear(channels[idx], channels[idx + 1], bias=True))
            layers.append(nn.LayerNorm(channels[idx + 1]))
            layers.append(nn.ReLU(True))
        layers = layers[:-2]

        if layer_num == 0:
            layers = [torch.nn.Identity()]
        self.layers = nn.Sequential(*layers)

    def forward(self, decoder_input):
        output = self.layers(decoder_input)
        return output
