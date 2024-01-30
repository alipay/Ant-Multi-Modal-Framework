# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common import configurable
from antmmf.structures import NestedTensor


class DetrPositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    @configurable
    def __init__(
        self,
        num_pos_feats: int = 256,
        max_position_embeddings: int = 50,
    ):
        super().__init__()
        self.row_embed = nn.Embedding(max_position_embeddings, num_pos_feats)
        self.col_embed = nn.Embedding(max_position_embeddings, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)  # w
        j = torch.arange(h, device=x.device)  # h
        x_emb = self.col_embed(i)  # w, 256
        y_emb = self.row_embed(j)  # h, 256
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),  # h, w, 256
                    y_emb.unsqueeze(1).repeat(1, w, 1),  # h, w, 256
                ],
                dim=-1,
            )
            .permute(2, 0, 1)  # 512, h, w
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)  # b, 512, h, w
        )
        return pos
