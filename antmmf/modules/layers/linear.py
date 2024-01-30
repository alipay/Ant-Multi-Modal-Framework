# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


class Linear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        weight_norm_dim: int = -1,
    ):
        super(Linear, self).__init__()
        self.lc = weight_norm(
            nn.Linear(in_features=in_dim, out_features=out_dim), dim=weight_norm_dim
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.lc(x)


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(out_dim, in_dim), requires_grad=True
        )
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """
        normalize input and weight for linear
        """
        norm_x = F.normalize(x, p=2, dim=-1)
        norm_w = F.normalize(self.weight, p=2, dim=-1)
        y = F.linear(norm_x, norm_w)
        return norm_x, y


class LinReLU(torch.nn.Module):
    __constants__ = ["bias"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(LinReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.nn.parameter.Parameter(
            torch.Tensor(in_features, out_features)
        )
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weights)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = (inputs - self.bias) @ self.weights
        output = F.relu(output)

        return output

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"
