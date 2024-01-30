# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import torch.nn.functional as F
from torch import nn

from .graph_encoder import GraphEncoder


@GraphEncoder.register()
class GATEncoder(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super(GATEncoder, self).__init__()

        self.dropout = dropout
        layers = nn.ModuleList()

        from torch_geometric.nn import GATConv

        for i in range(num_layers - 1):
            layers.append(GATConv(dim_in, hidden_size, num_heads, dropout=self.dropout))
            dim_in = hidden_size * num_heads

        layers.append(
            GATConv(dim_in, dim_out, heads=1, concat=False, dropout=self.dropout)
        )

        self.layers = layers

    def forward(self, data):

        x = data.x
        if not isinstance(x, torch.Tensor):
            x = x.to_dense()
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, data.edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, data.edge_index)
        return x
