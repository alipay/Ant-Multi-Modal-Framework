# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common.registry import registry
from antmmf.modules.classifier import ClassifierLayer


@ClassifierLayer.register()
class LanguageDecoder(nn.Module):
    """
    TODO: Add document here.
    """

    def __init__(self, in_dim, out_dim, hidden_dim, dropout, fc_bias_init):
        super().__init__()

        self.language_lstm = nn.LSTMCell(in_dim + hidden_dim, hidden_dim, bias=True)
        self.fc = nn.utils.weight_norm(nn.Linear(hidden_dim, out_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights(fc_bias_init)

    def init_weights(self, fc_bias_init):
        self.fc.bias.data.fill_(fc_bias_init)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, weighted_attn):
        # TODO: make state as argument of forward, instead of using registry.
        state = registry.get("{}_lstm_state".format(weighted_attn.device))
        h1, c1 = state["td_hidden"]
        h2, c2 = state["lm_hidden"]

        # Language LSTM
        h2, c2 = self.language_lstm(torch.cat([weighted_attn, h1], dim=1), (h2, c2))
        predictions = self.fc(self.dropout(h2))

        # Update hidden state for t+1
        state["lm_hidden"] = (h2, c2)

        return predictions
