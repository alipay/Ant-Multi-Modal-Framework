# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import logging
import math
from typing import Union, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, LayerNorm
from torch_geometric.typing import PairTensor

from antmmf.modules.message_passing import MessagePassing
from antmmf.modules.utils import ccorr


class DeltaConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attn_heads: int = 1,
        residual_beta: float = None,
        learn_beta: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        trans_method: str = "add",
        edge_fusion_mode: str = "add",
        time_fusion_mode: str = None,
        head_fusion_mode: str = "avg",
        residual_fusion_mode: str = None,
        edge_dim: int = None,
        rel_embed_dim: int = None,
        time_embed_dim: int = 0,
        attention_type: str = "GO",
        with_edge_feature: bool = False,
        norm_type: str = "LayerNorm",
        aggregation: str = "add",
        sparse_softmax_norm: bool = True,
        **kwargs,
    ):
        from kgrl.models.pytorch.layers import (
            GraphNorm,
            MXAttention,
            SparseLinear,
        )

        super(DeltaConv, self).__init__(aggr=aggregation, node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_heads = attn_heads

        self.residual_beta = residual_beta
        self.learn_beta = learn_beta
        self.dropout = dropout

        self.edge_dim = edge_dim

        self.head_fusion_mode = head_fusion_mode
        self.norm_type = norm_type
        self.trans_method = trans_method
        self.attention_type = attention_type

        self.sparse_softmax_norm = sparse_softmax_norm

        if not with_edge_feature:
            logging.info("with_edge_feature is False, set edge_dim to None !")
            self.edge_dim = None
        else:
            self.lin_edge = SparseLinear(
                self.edge_dim, attn_heads * out_channels, bias=False
            )

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # _check_attn_dim
        attn_in_dim = in_channels[0]
        attn_out_dim = (
            out_channels * self.attn_heads
            if self.head_fusion_mode == "concat"
            else out_channels
        )
        if self.trans_method == "concat":
            attn_in_dim += rel_embed_dim
        else:
            assert attn_in_dim == rel_embed_dim, (
                f"[Delta >> Translation Error] Node embedding dimension {attn_in_dim} is not equal with relation "
                f"embedding dimension {rel_embed_dim} when you are using '{trans_method}' translation method."
            )

        if time_fusion_mode == "concat":
            attn_in_dim += time_embed_dim
        elif time_fusion_mode is not None:
            assert attn_in_dim == time_embed_dim, (
                f"[Delta >> Time Fusion Error] Time embedding dimension {time_embed_dim} is not equal with"
                f" edge fusion result embedding dimension {attn_in_dim} when you are "
                f"using '{time_fusion_mode}' time fusion mode."
            )

        self.lin_key = Linear(attn_in_dim, attn_heads * out_channels)
        self.lin_query = Linear(in_channels[1], attn_heads * out_channels)
        self.lin_value = Linear(attn_in_dim, attn_heads * out_channels)

        def process_fusion_mode(fusion_mode, candidates):
            return None if fusion_mode not in candidates else fusion_mode

        residual_fusion_mode = process_fusion_mode(
            residual_fusion_mode, ["concat", "add"]
        )
        edge_fusion_mode = process_fusion_mode(edge_fusion_mode, ["concat"])
        time_fusion_mode = process_fusion_mode(time_fusion_mode, ["concat"])

        if residual_fusion_mode == "concat":
            self.lin_ffn_0 = Linear(in_channels[1] + attn_out_dim, out_channels + 32)
            self.lin_ffn_1 = Linear(out_channels + 32, out_channels)
        elif residual_fusion_mode == "add":
            if head_fusion_mode == "concat":
                self.lin_ffn_1 = Linear(
                    attn_heads * out_channels, out_channels, bias=bias
                )

            out_dim = (
                attn_heads * out_channels
                if head_fusion_mode == "concat"
                else out_channels
            )
            self.lin_skip = Linear(in_channels[1], out_dim, bias=bias)
            if learn_beta:
                dim = (
                    3 * attn_heads * out_channels
                    if head_fusion_mode == "concat"
                    else 2 * out_channels
                )
                self.lin_beta = Linear(dim, 1, bias=False)
        else:
            self.lin_ffn_0 = Linear(attn_out_dim, out_channels + 32)
            self.lin_ffn_1 = Linear(out_channels + 32, out_channels)

        self.residual_fusion_func = {
            "concat": self.residual_fusion_concat,
            "add": self.residual_fusion_add,
            None: self.residual_fusion_none,
        }[residual_fusion_mode]

        self.edge_fusion_func = {
            "concat": lambda x_j, edge_attr: torch.cat([x_j, edge_attr], dim=-1),
            None: lambda x_j, edge_attr: x_j + edge_attr,
        }[edge_fusion_mode]

        self.time_fusion_func = {
            "concat": lambda x_j, edge_time_embed: torch.cat(
                [x_j, edge_time_embed], dim=-1
            ),
            None: lambda x_j, edge_time_embed: x_j + edge_time_embed,
        }[time_fusion_mode]

        self.head_fusion_preprocess = {
            True: lambda out: out.view(-1, attn_heads * out_channels),
            False: lambda out: out.mean(dim=1),
        }[head_fusion_mode is not None]

        if norm_type == "LayerNorm":
            self.norm_func = LayerNorm(out_channels)
        elif norm_type == "GraphNorm":
            self.norm_func = GraphNorm(out_channels)
        if attention_type == "MX":
            self.mx_att = MXAttention(out_channels, attn_heads)

    def residual_fusion_concat(self, x: Union[Tuple, List, PairTensor], out: Tensor):
        out = torch.cat([out, x[1]], dim=-1)
        out = F.relu_(self.lin_ffn_0(out))
        out = self.lin_ffn_1(out)
        return out

    def residual_fusion_add(self, x: Union[Tuple, List, PairTensor], out: Tensor):
        x_skip = self.lin_skip(x[1])
        if self.learn_beta:
            beta = self.lin_beta(torch.cat([out, x_skip], dim=-1))
            beta = beta.sigmoid()
            out = beta * x_skip + (1 - beta) * out
        else:
            if self.residual_beta is not None:
                out = self.residual_beta * x_skip + (1 - self.residual_beta) * out
            else:
                out = out + x_skip
        if self.head_fusion_mode == "concat":
            out = self.lin_ffn_1(out)
        return out

    def residual_fusion_none(self, x: Union[Tuple, List, PairTensor], out: Tensor):
        out = F.relu_(self.lin_ffn_0(out))
        out = self.lin_ffn_1(out)
        return out

    def forward(
        self,
        x,
        edge_index,
        edge_type,
        edge_attr,
        edge_time_embed,
        rel_embed,
        num_nodes,
    ):

        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            edge_time_embed=edge_time_embed,
            edge_type=edge_type,
            rel_embed=rel_embed,
            size=None,
        )
        out = self.head_fusion_preprocess(out)
        out = self.residual_fusion_func(x, out)

        if self.norm_type == "LayerNorm":
            out = self.norm_func(out)
        elif self.norm_type == "GraphNorm":
            out = self.norm_func(out, num_nodes)
        return out

    def message(
        self,
        x_i,
        x_j,
        edge_attr,
        edge_time_embed,
        edge_type,
        rel_embed,
        index,
        ptr,
        size_i,
    ):
        from kgrl.models.pytorch.utils import sparse_softmax

        edge_type_embed = rel_embed.index_select(0, edge_type.flatten())
        x_j = self.rel_transform(x_j, edge_type_embed)

        if self.edge_dim is not None:
            assert (
                edge_attr is not None
            ), "[Delta >> Edge Fusion Error] Edge feature should be given."

            x_j = self.edge_fusion_func(x_j, edge_attr)
            edge_attr = self.lin_edge(edge_attr).view(
                -1, self.attn_heads, self.out_channels
            )

        if edge_time_embed is not None:
            x_j = self.time_fusion_func(x_j, edge_time_embed)

        # [batch_size, attn_heads, out_channels]
        query = self.lin_query(x_i).view(-1, self.attn_heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.attn_heads, self.out_channels)
        value = self.lin_value(x_j).view(-1, self.attn_heads, self.out_channels)

        if self.edge_dim is not None:
            value = value + edge_attr

        if self.attention_type == "MX":
            alpha = self.mx_att(index, query, key, size_i)
        else:
            alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = sparse_softmax(
            alpha, index, ptr, size_i, normalize=self.sparse_softmax_norm
        )
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        value = value * alpha.view(-1, self.attn_heads, 1)
        return value

    def rel_transform(self, ent_embed, edge_type_embed):
        if self.trans_method == "corr":
            trans_embed = ccorr(ent_embed, edge_type_embed)
        elif self.trans_method == "sub":
            trans_embed = ent_embed - edge_type_embed
        elif self.trans_method == "mult":
            trans_embed = ent_embed * edge_type_embed
        elif self.trans_method == "add":
            trans_embed = ent_embed + edge_type_embed
        elif self.trans_method == "concat":
            trans_embed = torch.cat([ent_embed, edge_type_embed], dim=1)
        else:
            raise NotImplementedError
        return trans_embed
