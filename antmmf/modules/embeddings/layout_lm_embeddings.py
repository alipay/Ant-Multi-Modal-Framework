# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from typing import Optional
import torch
from torch import nn
from antmmf.common import configurable


class LayoutLMEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    @configurable
    def __init__(
        self,
        vocab_size: int = 21128,
        hidden_size: int = 768,
        max_position_embeddings: int = 256,  # sequential length
        max_2d_position_embeddings: int = 1024,  # 2d position length
        type_vocab_size: int = 2,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-6,
        embeddings: Optional[nn.Module] = None,
    ):
        super(LayoutLMEmbeddings, self).__init__()
        if embeddings is None:
            self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.position_embeddings = nn.Embedding(  # Embedding(512, 768)
                max_position_embeddings,
                hidden_size,  # max_position_embeddings = 512
            )
            self.token_type_embeddings = nn.Embedding(
                type_vocab_size, hidden_size
            )  # type_vocab_size = 2
        else:
            self.position_embeddings = embeddings.position_embeddings
            self.token_type_embeddings = embeddings.token_type_embeddings
            self.word_embeddings = embeddings.word_embeddings

        self.max_2d_position_embeddings = max_2d_position_embeddings
        if max_2d_position_embeddings > 0:
            if hasattr(embeddings, "x_position_embeddings"):
                self.x_position_embeddings = embeddings.x_position_embeddings
            else:
                self.x_position_embeddings = nn.Embedding(  # Embedding(1024, 768)
                    max_2d_position_embeddings,
                    hidden_size
                    # max_2d_position_embeddings = 1024   hidden_size=768
                )
            if hasattr(embeddings, "y_position_embeddings"):
                self.y_position_embeddings = embeddings.y_position_embeddings
            else:
                self.y_position_embeddings = nn.Embedding(
                    max_2d_position_embeddings, hidden_size
                )
            if hasattr(embeddings, "h_position_embeddings"):
                self.h_position_embeddings = embeddings.h_position_embeddings
            else:
                self.h_position_embeddings = nn.Embedding(
                    max_2d_position_embeddings, hidden_size
                )
            if hasattr(embeddings, "w_position_embeddings"):
                self.w_position_embeddings = embeddings.w_position_embeddings
            else:
                self.w_position_embeddings = nn.Embedding(
                    max_2d_position_embeddings, hidden_size
                )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_2d_position_embedding(self, bbox):
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        h_position_embeddings = self.h_position_embeddings(
            bbox[:, :, 3] - bbox[:, :, 1]
        )
        w_position_embeddings = self.w_position_embeddings(
            bbox[:, :, 2] - bbox[:, :, 0]
        )  # torch.Size([9, 512, 768])
        position_2d_embeddings = (
            left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
        )  # torch.Size([9, 512, 768])
        return position_2d_embeddings

    def get_1d_position_embedding(self, input_ids=None, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1),
                dtype=torch.long,
                device=input_ids.device,  # torch.Size([512])
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return self.position_embeddings(position_ids)  # torch.Size([9, 512, 768])

    def forward(
        self,
        input_ids,
        bbox,
        token_type_ids=0,
        position_ids=None,
        return_position_embed=False,
    ):
        if isinstance(token_type_ids, int):
            token_type_ids = torch.zeros_like(input_ids).fill_(token_type_ids)

        words_embeddings = self.word_embeddings(input_ids)  # torch.Size([9, 512, 768])
        position_embeddings = self.get_1d_position_embedding(
            input_ids=input_ids, position_ids=position_ids
        )  # torch.Size([9, 512, 768])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.max_2d_position_embeddings > 0:
            position_embeddings += self.get_2d_position_embedding(bbox)

        embeddings = (
            words_embeddings + position_embeddings + token_type_embeddings
        )  # torch.Size([9, 512, 768])
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return (
            (embeddings, position_embeddings) if return_position_embed else embeddings
        )  # torch.Size([9, 512, 768])
