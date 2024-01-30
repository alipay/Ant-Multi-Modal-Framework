# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn


class ConditionalLayerNorm(nn.Module):
    """This module implements a conditional layer normalization, reference:
       http://www.sniper97.cn/index.php/note/deep-learning/note-deep-learning/3782/，
       This module works on joint models, fusing the results of the previous model into the input of the next model.

    Args:
            input_hidden_dim (`scalar`): The dimension of the data that needs to be normalized.
            layer_norm_cond_size (`scalar`): The size of the input condition.
            eps (`scalar`): it is a scalar.
            inputs (`~torch.Tensor`): input tensor of size ``(batch_size, seq_length, hidden_dim)``.
            layer_norm_cond (`~torch.Tensor`):  condition tensor of size ``(batch_size, seq_length, hidden_dim)``.

    Returns:
        `~torch.Tensor`: output tensor of size ``(batch_size, seq_length, hidden_dim)``.

    """

    def __init__(self, input_hidden_dim, layer_norm_cond_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.Tensor(input_hidden_dim))
        self.beta = nn.Parameter(torch.Tensor(input_hidden_dim))
        self.gamma_dense = nn.Linear(layer_norm_cond_size, input_hidden_dim, bias=False)
        self.beta_dense = nn.Linear(layer_norm_cond_size, input_hidden_dim, bias=False)
        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        nn.init.zeros_(self.gamma_dense.weight)
        nn.init.zeros_(self.beta_dense.weight)
        nn.init.trunc_normal_(self.gamma)
        nn.init.trunc_normal_(self.beta)

    def forward(self, inputs, layer_norm_cond=None):
        assert (
            layer_norm_cond is not None
        ), "Conditional tensor need to input when use conditional layer norm"
        input_dim = inputs.dim()
        cond_dim = layer_norm_cond.dim()
        assert (
            input_dim == cond_dim
        ), "inputs and layer_norm_cond dim must equals inputs dim, the shape like (batch_size, seq_length, hidden_dim)"
        # (batch_size, seq_length, input_dim)
        gamma = self.gamma_dense(layer_norm_cond) + self.gamma
        # (batch_size, seq_length, input_dim)
        beta = self.beta_dense(layer_norm_cond) + self.beta
        # (batch_size, seq_length, input_dim)
        mean = torch.mean(inputs, dim=-1, keepdim=True)
        std = torch.std(inputs, dim=-1, keepdim=True)
        outputs = gamma * (inputs - mean) / (std + self.eps) + beta
        return outputs
