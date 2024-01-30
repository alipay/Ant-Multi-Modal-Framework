# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from typing import Optional
from torch import nn
from antmmf.common import configurable


class ClipVisualEmbedding(nn.Module):
    """
    Visual Embedding for clip-BERT, which takes input of both image and video (multi-frame).
    Copy from:
    https://github.com/jayleicn/ClipBERT/blob/main/src/modeling/modeling.py#L40-L153

    Refer to:
    [1] Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling
    [https://arxiv.org/abs/2102.06183]
    [2] Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers
    [https://arxiv.org/abs/2004.00849]
    """

    @configurable
    def __init__(
        self,
        max_position_embeddings: int = 50,
        num_pos_feats: int = 256,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        pixel_random_sampling_size: Optional[int] = None,
    ):
        super(ClipVisualEmbedding, self).__init__()

        self.pixel_random_sampling_size = pixel_random_sampling_size

        self.row_position_embeddings = nn.Embedding(
            max_position_embeddings, num_pos_feats
        )
        self.col_position_embeddings = nn.Embedding(
            max_position_embeddings, num_pos_feats
        )
        self.token_type_embeddings = nn.Embedding(1, num_pos_feats)
        self.LayerNorm = nn.LayerNorm(num_pos_feats, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, grid):
        """
        Args:
            grid: (B, n_frm, H, W, C), note that #frm can be 1
        Returns:
        """

        bsz, _, _, _, hsz = grid.shape

        # temporal mean pooling
        grid = grid.mean(1)  # (B, H, W, d)
        grid += self.get_2d_positional_embeddings(grid)  # (B, H, W, d)
        # image token sequence
        visual_tokens = grid.view(bsz, -1, hsz)  # (B, H*W, d)

        # perform random sampling. It is only used in training phase
        # of pre-training, but not used in inference or downstream tasks.
        # see detail at Table 6 in paper:
        # Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers
        # Randomly pixel sampling method can contribute 0:5 score on VQA, about 2:0
        # score on retrieval tasks and 0:4 score on NLVR2.
        sampled_indices = torch.arange(
            visual_tokens.size(1), device=visual_tokens.device
        )
        if self.pixel_random_sampling_size and self.training:
            sampled_indices = ClipVisualEmbedding.get_random_sample_indices(
                seq_len=visual_tokens.shape[1],
                num_samples=self.pixel_random_sampling_size,
                device=visual_tokens.device,
            )
            visual_tokens = visual_tokens.index_select(
                dim=1, index=sampled_indices
            )  # (B, #samples, d)
        visual_tokens_shape = visual_tokens.shape[:-1]  # (B, H*W)
        device = visual_tokens.device

        # image token type embeddings.
        token_type_ids = torch.zeros(
            visual_tokens_shape, dtype=torch.long, device=device
        )
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = visual_tokens + position_embeddings + token_type_embeddings
        embeddings = visual_tokens + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, sampled_indices  # (B, H*W, d)

    @staticmethod
    def get_random_sample_indices(seq_len, num_samples=100, device=torch.device("cpu")):
        """
        Args:
            seq_len: int, the sampled indices will be in the range [0, seq_len-1]
            num_samples: sample size
            device: torch.device
        Returns:
            1D torch.LongTensor consisting of sorted sample indices
            (sort should not affect the results as we use transformers)
        """
        if num_samples >= seq_len:
            # return all indices
            sample_indices = torch.arange(seq_len, device=device)
        else:
            seq_indices = torch.arange(seq_len, device=device)
            p = torch.tensor([1.0] * seq_len)
            idx = p.multinomial(num_samples=seq_len, replacement=False)
            idx = idx[:num_samples]
            sample_indices, _ = seq_indices[idx].sort()
        return sample_indices

    def get_2d_positional_embeddings(self, grid):
        """
        Args:
            grid: (B, *, H, W, d)
        Returns:
            (B, *, H, W, d)
        """
        height, width, hsz = grid.shape[-3:]

        # add row-wise position embeddings
        row_position_ids = torch.arange(
            height, dtype=torch.long, device=grid.device
        )  # (H, )
        row_position_embeddings = self.row_position_embeddings(
            row_position_ids
        )  # (H, d)
        row_shape = (1,) * (len(grid.shape) - 3) + (height, 1, hsz)  # (1, *1, H, 1, d)
        row_position_embeddings = row_position_embeddings.view(*row_shape).expand_as(
            grid
        )

        # add column-wise position embeddings
        col_position_ids = torch.arange(
            width, dtype=torch.long, device=grid.device
        )  # (W, )
        col_position_embeddings = self.col_position_embeddings(
            col_position_ids
        )  # (W, d)
        col_shape = (1,) * (len(grid.shape) - 3) + (1, width, hsz)  # (1, *1, 1, W, d)
        col_position_embeddings = col_position_embeddings.view(*col_shape).expand_as(
            grid
        )

        return row_position_embeddings + col_position_embeddings
