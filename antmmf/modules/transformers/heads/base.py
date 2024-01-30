# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from typing import Dict, Any
from torch import nn


class PredictableHead(nn.Module):
    """
    Base class for transformer predictable heads that support
    inference mode.
    """

    def forward_head(self, encoder_output, decoder_output, **kwargs):
        raise NotImplementedError

    def get_loss_metric(self, predictions, targets) -> Dict[str, Any]:
        raise NotImplementedError

    def forward(self, encoder_output=None, decoder_output=None, targets=None, **kwargs):
        predictions = self.forward_head(
            encoder_output=encoder_output, decoder_output=decoder_output, **kwargs
        )
        return self.get_loss_metric(predictions, targets)
