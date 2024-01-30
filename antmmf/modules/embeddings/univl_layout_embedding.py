# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from typing import Optional
from torch import nn
from antmmf.common import Configuration, configurable
from .layout_lm_embeddings import LayoutLMEmbeddings


class UnivlLayoutEmbedding(LayoutLMEmbeddings):
    """
    refer to:
    layoutLMv2:
    https://huggingface.co/transformers/_modules/transformers/models/layoutlmv2/modeling_layoutlmv2.html#LayoutLMv2Model
    """

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
        v_hidden_size: int = 768,
        has_visual_segment_embedding: bool = False,
    ):
        # delete the unused arguments
        config = Configuration(locals())
        for key in [
            "self",
            "__class__",
            "v_hidden_size",
            "has_visual_segment_embedding",
        ]:
            del config[key]

        super(UnivlLayoutEmbedding, self).__init__(**config)

        self.v_LayerNorm = nn.LayerNorm(v_hidden_size, eps=layer_norm_eps)
        if has_visual_segment_embedding:
            self.visual_segment_embedding = nn.Parameter(
                nn.Embedding(1, v_hidden_size).weight[0]
            )

        self.has_visual_segment_embedding = has_visual_segment_embedding

    def _calc_img_embeddings(
        self,
        visual_tokens,
        position_2d_embeddings=None,
        position_1d_embeddings=None,
        bbox=None,
        position_ids=None,
    ):
        """

        :param visual_tokens: B, seq, h
        :param position_2d_embeddings: B, seq, h
        :param position_1d_embeddings: B, seq, h
        :param bbox: B, seq, 4
        :param position_ids: B, seq
        :return:
        """
        if position_2d_embeddings is None:
            position_2d_embeddings = self.get_2d_position_embedding(bbox)
        if position_1d_embeddings is None:
            position_1d_embeddings = self.get_1d_position_embedding(
                position_ids=position_ids
            )

        embeddings = visual_tokens + position_1d_embeddings + position_2d_embeddings

        # image token type embeddings.
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding

        embeddings = self.v_LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, (position_1d_embeddings, position_2d_embeddings)

    def _calc_text_embeddings(
        self,
        text_tokens,
        position_2d_embeddings=None,
        position_1d_embeddings=None,
        bbox=None,
        position_ids=None,
        segment_id=None,
    ):
        if position_1d_embeddings is None:
            position_1d_embeddings = self.get_1d_position_embedding(
                position_ids=position_ids
            )

        if position_2d_embeddings is None:
            position_2d_embeddings = self.get_2d_position_embedding(bbox)

        embeddings = text_tokens + position_1d_embeddings + position_2d_embeddings
        if segment_id is not None:
            token_type_ids = torch.zeros(
                text_tokens.shape[:2], dtype=torch.long, device=text_tokens.device
            ).fill_(segment_id)
            embeddings += self.token_type_embeddings(token_type_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, (position_1d_embeddings, position_2d_embeddings)
