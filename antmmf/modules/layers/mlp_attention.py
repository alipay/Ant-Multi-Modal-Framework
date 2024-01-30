# Copyright (c) 2023 Ant Group and its affiliates.

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MLPAttention(nn.Module):
    """
    attention mechanism proposed in https://arxiv.org/pdf/1409.0473.pdf
    """

    def __init__(self, hidden_size, key_size, query_size):
        super(MLPAttention, self).__init__()
        self.key_layer = nn.Linear(key_size, hidden_size)
        self.query_layer = nn.Linear(query_size, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)

    def forward(self, query, values, mask=None):
        """

        Args:
            query ('~torch.Tensor'): query tensor to calculate the attention weights
                "(batch_size, 1, query_size)"
            values ('~torch.Tensor'): key and value tensor for attention weights and
                context vector calculation, "(batch_size, seq_length, key_size)"
            mask ('~torch.Tensor'): mask out keys position (0 in invalid positions, 1 else),
                "(batch_size, 1, seq_length)"

        Returns:
            `~torch.Tensor`: context vector of shape "(batch_size, 1, hidden_size)",
            `~torch.Tensor`: attention weights of shape "(batch_size, 1, seq_length)"
        """
        query = self.query_layer(query)
        values = self.key_layer(values)
        scores = self.energy_layer(torch.tanh(query + values)).squeeze(2)
        if mask is not None:
            scores = torch.where(mask > 0, scores, scores.new_full([1], -np.inf))
        attn_weights = F.softmax(scores, dim=-1).unsqueeze(1)
        context = attn_weights @ values
        return context, attn_weights
