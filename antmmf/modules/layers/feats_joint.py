# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from torch import nn
from typing import Union, Callable
from transformers.activations import get_activation


class ActivationWrapper(nn.Module):
    def __init__(self, func: Callable):
        super(ActivationWrapper, self).__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Joint(nn.Module):
    """Multiple Layer Perceptron.

    Args:
        in_dim (int): dimension of input.
        out_dim (int): dimension of output.
        hidden_dim (int): dimension of hidden layer(s), its default value is in_dim.
        num_layers (int): the number of hidden layer(s), default value is 0.
        dropout (float): dropout ratio, it works only when it is set to a float in the range from 0 to 1, default None.
        activation (Union[str, nn.Module, Callable]): activation function. If it is a string, then we will map it into
          the function of torch according to :func:`transformers.activations.get_activation`. Moreover, it can also be a
          :external:py:class:`Module <torch.nn.Module>`, callable function, or `None`, and its default value is "relu".
        batch_norm (bool): whether use BatchNorm1d or not, default value is True.
        weight_norm (bool): apply :external:py:func:`weight_norm <torch.nn.utils.weight_norm>` on the layer(s) Linear,
          default value is False.

    """

    def __init__(
        self,
        fc_layer_num: int,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = None,
        num_layers: int = 0,
        dropout: float = 0.5,
        activation: Union[str, nn.Module, Callable] = "relu",
        batch_norm: bool = True,
        weight_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim

        if isinstance(activation, str):
            activation = get_activation(activation)
        if activation is not None and not isinstance(activation, nn.Module):
            activation = ActivationWrapper(activation)

        layers = []
        for _ in range(num_layers):
            layer = nn.Linear(in_dim, hidden_dim)
            if weight_norm:
                layer = nn.utils.weight_norm(layer, dim=-1)
            layers.append(layer)

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation is not None:
                layers.append(activation)

            if dropout is not None and (1.0 >= dropout >= 0.0):
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layer = nn.Linear(in_dim, out_dim)
        if weight_norm:
            layer = nn.utils.weight_norm(layer, dim=-1)
        layers.append(layer)
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)
