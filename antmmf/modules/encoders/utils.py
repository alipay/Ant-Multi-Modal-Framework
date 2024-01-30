# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from torch import nn


class ModelForPretrainingMixin:
    """
    handle initialization for bert-based pretraining
    """

    def init_weights(self, module: nn.Module):
        """
        Initializes and prunes weights if needed.
        """
        # Initialize head weights, encoder weights initialized with BERT weights
        # during pretraining
        module.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
