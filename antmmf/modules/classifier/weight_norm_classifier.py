# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from torch import nn
from antmmf.modules.classifier import ClassifierLayer
from antmmf.modules.layers.mlp import MLP


@ClassifierLayer.register()
class WeightNormClassifier(MLP):
    """A MLP with two Linear Layer. Furthermore, weight norm is enabled while batch_norm is not.

    .. warning::

        This Module may will be removed in future, we suggest that directly use MLP instead.
    """

    def __init__(self, in_dim, out_dim, hidden_dim, dropout):
        super(WeightNormClassifier, self).__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=1,
            activation=nn.ReLU(),
            dropout=dropout,
            batch_norm=False,
            weight_norm=True,
        )
