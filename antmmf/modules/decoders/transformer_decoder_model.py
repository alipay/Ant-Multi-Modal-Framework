# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.modules.utils import get_mask


class TransformerDecoderModel(nn.Module):
    """
    TODO: add document here.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        nhead: int = 8,
        num_layers: int = 16,
        dropout: float = 0.5,
        nbr_queries: int = 1,
        **kwargs,
    ):
        super(TransformerDecoderModel, self).__init__()

        self.n_query = nbr_queries
        # inspired from work
        # End-to-End Object Detection with Transformers
        # by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko # noqa
        # ECCV 2020
        self.query = torch.randn((self.n_query, d_model))
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)

        self.output_proj = nn.Linear(d_model, num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        init_range = 0.1
        self.query.data.uniform_(-init_range, init_range)
        self.output_proj.weight.data.uniform_(-init_range, init_range)
        self.output_proj.bias.data.uniform_(-init_range, init_range)

    def forward(self, encoder_output, src_length, decoder_query=None):
        """
        Input：
        encoder_output: encoder的输出, shape [batch size, source length, source dimension]
        src_length: source sentence的长度, shape [bsz]
        decoder_query: 这个decoder支持multiple queries，可以从外部带入或者内部学习
                       如果没有给定，则内部根据在 init 时设定的number of queries 来决定

        Output:
            scores, shape in [batch size , dimension]
            注意这个dimension是有 number_queries * number_class 来决定的
            由此可以做multiple object detection等图像检测任务，或者文本中的多目标任务
        """
        # binary mask of valid source vs padding
        bsz, length, _ = encoder_output.size()

        # binary mask of valid source vs padding
        src_mask = (
            (1.0 - get_mask(src_length, length))
            .type(torch.bool)
            .to(encoder_output.device)
        )

        self.query = self.query.to(encoder_output.device)
        query = (
            self.query.unsqueeze(0).repeat(bsz, 1, 1)
            if decoder_query is None
            else decoder_query
        )
        # encoder_output [bsz, length, dim]
        encoder_output = encoder_output.transpose(0, 1)
        query = query.transpose(0, 1)
        output = self.transformer_decoder(
            memory=encoder_output, tgt=query, memory_key_padding_mask=src_mask
        )
        output = output.transpose(0, 1)

        output = self.output_proj(output)
        return output
