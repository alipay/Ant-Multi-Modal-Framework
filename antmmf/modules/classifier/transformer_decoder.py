# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.modules.classifier import ClassifierLayer
from antmmf.modules.decoders import TransformerDecoderModel


@ClassifierLayer.register()
class TransformerDecoderForClassificationHead(nn.Module):
    """
    this uses the above TransformerDecoderModel on snapshot
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.5,
        nbr_queries: int = 1,
        **kwargs,
    ):
        super(TransformerDecoderForClassificationHead, self).__init__()
        self.module = TransformerDecoderModel(
            d_model, num_classes, nhead, num_layers, dropout, nbr_queries, **kwargs
        )

    def forward(self, x):
        bsz, dim = x.size()
        inputs = x.reshape(bsz, 1, dim)
        output = self.module(inputs, torch.ones((bsz,), dtype=torch.long))
        output = output.view(bsz, -1)
        return output
