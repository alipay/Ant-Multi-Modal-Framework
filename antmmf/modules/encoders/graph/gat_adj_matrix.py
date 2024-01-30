# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import torch.nn.functional as F
from torch import nn
from .graph_encoder import GraphEncoder


class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer for building GAT network.
    Version for adjacency matrix.

    Args:
        in_features(int): dimension of input tensor.
        out_features(int): dimension of output tensor.
        dropout(float): dropout probability in forward calculation.
        alpha(float): param for leakyrelu.
        concat(bool): if 'true': output tensor should be activated by elu function.

    References:
        [1]: https://github.com/Diego999/pyGAT/blob/master/models.py
        [2]: https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        """
        input([batch_size, N, in_features]): input tensor
        adj([batch_size, N, N]): adjacency matrix
        """
        h = self.W(input)  # [batch_size, N, out_features]
        batch_size, N, out_features = h.size()
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, N)
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, N).transpose(1, 2)
        e = self.leakyrelu(middle_result1 + middle_result2)
        attention = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


@GraphEncoder.register()
class GAT_adj_matrix(nn.Module):
    """
    Note that this version is for adjacency matrix while 'GATEncoder' is for adjacency list.
    """

    def __init__(self, dim_in, dim_hid, dim_out, dropout, alpha, num_heads):
        super(GAT_adj_matrix, self).__init__()
        self.dropout = dropout
        self.attentions = [
            GraphAttentionLayer(
                dim_in, dim_hid, dropout=dropout, alpha=alpha, concat=True
            )
            for _ in range(num_heads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)
        self.out_att = GraphAttentionLayer(
            dim_hid * num_heads, dim_out, dropout=dropout, alpha=alpha, concat=False
        )

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=2)
