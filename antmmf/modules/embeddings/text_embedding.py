# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from functools import lru_cache
from antmmf.common import configurable
from antmmf.modules.module_registry import ModuleRegistry


class TextEmbedding(ModuleRegistry):
    def __init__(
        self,
        type: str = None,
        emb_type: str = None,
        embedding_dim: int = None,
        **kwargs,
    ):
        emb_type = type or emb_type
        assert emb_type is not None, "type or emb_type should be set."
        super(TextEmbedding, self).__init__(emb_type, **kwargs)

        if not hasattr(self.module, "text_out_dim"):
            self.module.text_out_dim = embedding_dim
        self.text_out_dim = self.module.text_out_dim


TextEmbedding.register(nn.Identity)
TextEmbedding.register(nn.Embedding)


@TextEmbedding.register()
class PreExtractedEmbedding(nn.Module):
    @configurable
    def __init__(self, out_dim, base_path):
        super(PreExtractedEmbedding, self).__init__()
        self.text_out_dim = out_dim
        self.base_path = base_path
        self.cache = {}

    def forward(self, qids):
        embeddings = []
        for qid in qids:
            embeddings.append(self.get_item(qid))
        return torch.stack(embeddings, dim=0)

    @lru_cache(maxsize=5000)
    def get_item(self, qid):
        return np.load(os.path.join(self.base_path, str(qid.item()) + ".npy"))


@TextEmbedding.register()
class AttentionTextEmbedding(nn.Module):
    @configurable
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        num_layers,
        dropout,
        conv1_out,
        conv2_out,
        kernel_size,
        padding,
        bidirectional: bool = False,
    ):
        super(AttentionTextEmbedding, self).__init__()

        self.text_out_dim = hidden_dim * conv2_out

        self.recurrent_unit = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        conv1_out = conv1_out
        conv2_out = conv2_out
        kernel_size = kernel_size
        padding = padding

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        self.recurrent_unit.flatten_parameters()
        # self.recurrent_unit.flatten_parameters()
        lstm_out, _ = self.recurrent_unit(x)  # N * T * hidden_dim
        lstm_drop = self.dropout(lstm_out)  # N * T * hidden_dim
        lstm_reshape = lstm_drop.permute(0, 2, 1)  # N * hidden_dim * T

        qatt_conv1 = self.conv1(lstm_reshape)  # N x conv1_out x T
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        # Over last dim
        qtt_softmax = F.softmax(qatt_conv2, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature = torch.bmm(qtt_softmax, lstm_drop)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        return qtt_feature_concat


@TextEmbedding.register()
class BiLSTMTextEmbedding(nn.Module):
    @configurable
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        num_layers,
        dropout,
        bidirectional=False,
        rnn_type="GRU",
    ):
        super(BiLSTMTextEmbedding, self).__init__()
        self.text_out_dim = hidden_dim
        self.bidirectional = bidirectional

        if rnn_type == "LSTM":
            rnn_cls = nn.LSTM
        elif rnn_type == "GRU":
            rnn_cls = nn.GRU
        else:
            raise NotImplementedError("only LSTM and GRU are available for rnn_type")

        self.recurrent_encoder = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.recurrent_encoder(x)
        # Return last state
        if self.bidirectional:
            return out[:, -1]

        forward_ = out[:, -1, : self.num_hid]
        backward = out[:, 0, self.num_hid :]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        output, _ = self.recurrent_encoder(x)
        return output
