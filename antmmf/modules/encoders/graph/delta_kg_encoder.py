# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from typing import List
import torch.nn.functional as F
from torch import nn

from antmmf.common import configurable
from .graph_encoder import GraphEncoder


@GraphEncoder.register()
class DeltaKGEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        drop_ratio: float = 0.0,
        head_network: str = "linear",
        node_feature_dim: int = 128,
        edge_feature_dim: int = 256,
        node_embed_dim: int = 128,
        edge_embed_dim: int = 128,
        time_embed_dim: int = 128,
        with_edge_feature: bool = True,
        subgraph_fusion_mode: int = 0,
        trans_method: str = "add",
        rel_embed_dim: int = 128,
        activation: str = None,
        aggregate_func: str = "delta",
        num_layers: int = 1,
        num_rel: int = 10,
        emb_list: List = None,
        inverse_edges_mode: int = 3,
        attn_heads: int = 4,
        residual_beta: float = None,
        learn_beta: bool = False,
        edge_fusion_mode: str = "add",
        time_fusion_mode: str = None,
        head_fusion_mode: str = "concat",
        residual_fusion_mode: str = None,
        attention_type: str = "GO",
        norm_type: str = "LayerNorm",
        sparse_softmax_norm: bool = True,
        aggregation: str = "add",
    ):
        super(DeltaKGEncoder, self).__init__()

        from kgrl.models.pytorch.layers import SparseLinear, DCNMix
        from kgrl.models.pytorch.conf import AddInverseEdgesMode

        self.drop_ratio = drop_ratio
        self.subgraph_fusion_mode = subgraph_fusion_mode

        if head_network == "linear":
            self.node_transform = SparseLinear(node_feature_dim, node_embed_dim)
        elif head_network == "DCNMix":
            self.node_transform = DCNMix(
                node_feature_dim,
                node_embed_dim,
                cross_num=3,
                dnn_hidden_units=[256, 128, 128],
                dnn_use_bn=False,
            )

        # concat or add edge feature when calculate neighbor message

        if with_edge_feature:
            if trans_method == "concat":
                edge_embed_dim = edge_embed_dim + rel_embed_dim
            self.edge_transform = SparseLinear(edge_feature_dim, edge_embed_dim)

        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.RReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.Tanh()

        self.convs = nn.ModuleList()
        self.aggregator = aggregate_func
        error_msg = (
            f"We only support five aggregators - ('compgcn', 'gat', 'transformer', 'delta',"
            f" 'RelationWiseNormConv'), but you have {self.aggregator} as an aggregator."
        )
        assert self.aggregator in [
            "compgcn",
            "gat",
            "transformer",
            "delta",
            "RelationWiseNormConv",
        ], error_msg

        num_relation = num_rel
        if not isinstance(emb_list, list):
            if head_network not in ["linear", "DCNMix"]:
                emb_list = [node_feature_dim] + [node_embed_dim] * num_layers
            else:
                emb_list = [node_embed_dim] * (num_layers + 1)

        if inverse_edges_mode == AddInverseEdgesMode.WITH_INVERSE:
            num_relation = num_relation * 2

        for i in range(num_layers):
            args = {
                "in_channels": emb_list[i],
                "out_channels": emb_list[i + 1],
                "attn_heads": attn_heads,
                "residual_beta": residual_beta,
                "learn_beta": learn_beta,
                "dropout": self.drop_ratio,
                "bias": True,
                "trans_method": trans_method,
                "edge_fusion_mode": edge_fusion_mode,
                "time_fusion_mode": time_fusion_mode,
                "head_fusion_mode": head_fusion_mode,
                "residual_fusion_mode": residual_fusion_mode,
                "edge_dim": edge_embed_dim,
                "rel_embed_dim": rel_embed_dim,
                "time_embed_dim": time_embed_dim,
                "attention_type": attention_type,
                "with_edge_feature": with_edge_feature,
                "norm_type": norm_type,
                "sparse_softmax_norm": sparse_softmax_norm,
                "aggregation": aggregation,
            }
            if self.aggregator == "RelationWiseNormConv":
                args["num_rel"] = num_relation
                from antmmf.modules.message_passing import RelationWiseNormConv

                self.convs.append(RelationWiseNormConv(**args))
            elif self.aggregator == "delta":
                from antmmf.modules.message_passing import DeltaConv

                self.convs.append(DeltaConv(**args))
            elif self.aggregator == "gat":
                from torch_geometric.nn import GATConv

                self.convs.append(
                    GATConv(
                        args["in_channels"],
                        args["out_channels"],
                        heads=8,
                        dropout=args["dropout"],
                        concat=False,
                    )
                )
            else:
                raise Exception(
                    f"[Delta >> Aggregator Error] unknown aggregator: {self.aggregator}"
                )
        self.lin_rel = nn.Linear(rel_embed_dim, node_embed_dim)

    def forward(self, data):
        from kgrl.models.pytorch.conf import SubGraphFusionMode

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_type = data.edge_types.coo()[1]
        edge_time_embed = data.edge_time_embed if "edge_time_embed" in data else None
        rel_embed = data.rel_embed

        if hasattr(self, "node_transform"):
            x = self.node_transform(x)
        if hasattr(self, "edge_transform") and edge_attr is not None:
            edge_attr = self.edge_transform(edge_attr)
        elif edge_type is not None:
            edge_attr = rel_embed.index_select(0, edge_type.flatten())

        # add edge_tag in edge_attr when subgraphfusion using tag embedding
        if self.subgraph_fusion_mode == SubGraphFusionMode.ONLY_TAG_EMBEDDING:
            edge_attr += data.edge_tag_embed

        results = []

        for i in range(len(self.convs)):
            if self.aggregator == "gat":
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](
                    x,
                    edge_index,
                    edge_type,
                    edge_attr,
                    edge_time_embed,
                    rel_embed,
                    data.num_nodes,
                )
            x = self.activation(x)
            if self.drop_ratio > 0:
                x = F.dropout(x, p=self.drop_ratio, training=self.training)
            results.append(x)

        return results, self.lin_rel(rel_embed)
