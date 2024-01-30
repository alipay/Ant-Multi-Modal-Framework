# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from typing import Optional
import torch
from torch import nn
from antmmf.common import Configuration, configurable
from .layout_lm_embeddings import LayoutLMEmbeddings


class VisualLayoutEmbeddings(LayoutLMEmbeddings):
    """
    Encode bbox position with LayoutLM 2d positon embedding style, which is quite different from
    Uniter/VilBERT style.
    Visual Embedding for
    VilBERT: https://github.com/e-bug/volta/blob/main/volta/embeddings.py#L127
    Uniter: https://github.com/e-bug/volta/blob/main/volta/embeddings.py#L401
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
        visual_feature_size: int = 2048,
        embeddings: Optional[nn.Module] = None,
    ):
        config = Configuration(locals())
        del config["visual_feature_size"]

        super(VisualLayoutEmbeddings, self).__init__(config)
        self.image_embeddings = nn.Linear(visual_feature_size, hidden_size)

    def forward(self, visual_token, bbox, token_type_ids=0):
        """

        Args:
            visual_token(torch.Tensor): [batch_size, num_boxes, visual_feature_dim]
            bbox(torch.Tensor): [batch_size, num_boxes, 4]
            token_type_ids: [batch_size, num_boxes]

        Returns:

        """
        # token embeddings + token type + 2d position embeddings
        token_embedding = self.image_embeddings(
            visual_token
        )  # [batch_size, num_boxes, hidden_size]
        if isinstance(token_type_ids, int):
            token_type_ids = torch.zeros(
                (visual_token.size(0), visual_token.size(1)),
                device=visual_token.device,
                dtype=torch.long,
            ).fill_(token_type_ids)
        token_type_embeddings = self.token_type_embeddings(
            token_type_ids
        )  # [batch_size, num_boxes, hidden_size]

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
        embeddings = (
            token_embedding
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )  # torch.Size([9, 512, 768])

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # torch.Size([9, 512, 768])
