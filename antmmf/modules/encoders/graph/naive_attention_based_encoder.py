# -- coding: utf-8 --

import torch.nn.functional as F
from torch import nn

from antmmf.common import Configuration


class NaiveAttentionBasedEncoder(nn.Module):
    def __init__(self, encoder_config: Configuration):
        from kgrl.models.pytorch.layers import SparseLinear

        super(NaiveAttentionBasedEncoder, self).__init__()

        node_feature_dim = encoder_config.get("node_feature_dim", 128)
        edge_feature_dim = encoder_config.get("edge_feature_dim", 128)
        node_embed_dim = encoder_config.get("node_embed_dim", 128)
        edge_embed_dim = encoder_config.get("edge_embed_dim", 128)
        attn_heads = encoder_config.get("attn_heads", 4)
        num_layers = encoder_config.get("num_layers", 1)
        self.drop_ratio = encoder_config.get("dropout", 0.0)

        assert node_embed_dim == edge_embed_dim, (
            f"in NaiveAttentionBasedEncoder, node_embed_dim must be equal to "
            f"edge_embed_dim, but got {node_embed_dim} vs. {edge_embed_dim}"
        )
        assert num_layers > 0, f"num_layers must be greater 0, but got {num_layers}"

        self.node_transform = SparseLinear(node_feature_dim, node_embed_dim, bias=False)
        self.edge_transform = SparseLinear(edge_feature_dim, edge_embed_dim, bias=False)
        from antmmf.modules.message_passing import QKVGraphConv

        self.conv_modules = nn.ModuleList(
            [
                QKVGraphConv(node_embed_dim, node_embed_dim, attn_heads)
                for _ in range(num_layers)
            ]
        )

    def forward(self, input_data):
        from torch_geometric.data import Batch

        assert isinstance(input_data, Batch)
        x = input_data.x
        edge_index = input_data.edge_index
        edge_attr = input_data.edge_attr

        x = self.node_transform(x)
        edge_features = self.edge_transform(edge_attr)
        for conv in self.conv_modules:
            x = x + conv(x, edge_index, edge_features)
            x = F.relu_(x)
            if self.drop_ratio > 0:
                x = F.dropout(x, p=self.drop_ratio, training=self.training)

        return x
