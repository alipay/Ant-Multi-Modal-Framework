# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import math
from torch import nn, Tensor
import torch.nn.functional as F
from .message_passing import MessagePassing


class QKVGraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_heads: int,
        dropout: float = 0.0,
        aggregation: str = "add",
        norm_type: str = None,
        **kwargs,
    ):
        from kgrl.models.pytorch.layers import GraphNorm

        super(QKVGraphConv, self).__init__(aggr=aggregation, node_dim=0, **kwargs)
        assert out_channels % attention_heads == 0, (
            f"out_channels must be an integer multiply of attention_heads, "
            f"but got {out_channels} vs. {attention_heads}"
        )
        assert (
            0.0 <= dropout < 1.0
        ), f"dropout must in range [0., 1.), but got {dropout}"
        self.attention_heads = attention_heads
        self.out_channels = out_channels
        self.dropout = dropout
        self.norm_type = norm_type

        self.lin_query = nn.Linear(in_channels, out_channels)
        self.lin_key = nn.Linear(in_channels, out_channels)
        self.lin_value = nn.Linear(in_channels, out_channels)

        if norm_type is not None:
            assert norm_type in ["LayerNorm", "GraphNorm"], (
                f"LayerNorm and GraphNorm are available for norm_type, "
                f"but got {norm_type}"
            )
            self.norm_func = {"LayerNorm": nn.LayerNorm, "GraphNorm": GraphNorm}[
                norm_type
            ](out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_feature: Tensor,
        num_nodes_per_sample: Tensor = None,
    ):
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, size=None, edge_feature=edge_feature)
        if self.norm_type == "LayerNorm":
            out = self.norm_func(out)
        if self.norm_type == "GraphNorm":
            out = self.norm_func(out, num_nodes_per_sample)
        return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_feature: Tensor,
        index,
        ptr,
        size_i,
    ) -> Tensor:
        """
        :param x_i: tail node features
        :param x_j: head node features
        :param edge_feature: features which contain edge_type and edge attr features, furthermore,
            it was selected by edge_type, so the first dimension is equal to that of x_i or x_j
        :param index: tail node index
        :param ptr:
        :param size_i:
        :return:
        """

        from kgrl.models.pytorch.utils import sparse_softmax

        out_channels = self.out_channels // self.attention_heads
        x_j = x_j + edge_feature
        query = self.lin_query(x_i).view(-1, self.attention_heads, out_channels)
        key = self.lin_key(x_j).view(-1, self.attention_heads, out_channels)
        value = self.lin_value(x_j).view(-1, self.attention_heads, out_channels)

        alpha = (query * key).sum(dim=-1) / math.sqrt(out_channels)
        alpha = sparse_softmax(alpha, index, ptr, size_i, normalize=True)
        if self.dropout > 0.0:
            F.dropout(alpha, p=self.dropout, training=self.training)
        value = value * alpha.view(-1, self.attention_heads, 1)
        return value.reshape(-1, self.out_channels)
