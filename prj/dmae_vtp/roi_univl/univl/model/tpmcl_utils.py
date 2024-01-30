# -*- coding: utf-8 -*-
# Copyright (c) 2023 Ant Group and its affiliates.
import torch
from torch import nn

class LinearXWeightPredictor(nn.Module):
    __constants__ = ['num_frames', 'num_tokens', 'embed_dim', 'qdim', 'kdim']

    def __init__(self, num_frames: int, num_tokens: int, embed_dim: int, qk_bias: bool = False,
                 qdim: int = None, kdim: int = None):
        super().__init__()
        self.num_frames = num_frames
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self._qk_same_embed_dim = self.qdim == embed_dim and self.kdim == embed_dim

        self.q_proj = nn.Linear(self.qdim, embed_dim, bias=qk_bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=qk_bias)
        self.qk_proj = nn.Linear(self.num_frames, self.num_tokens, bias=qk_bias)

        self.attn_proj = nn.Sequential(
            nn.LayerNorm([num_tokens, embed_dim * 2]),
            nn.Linear(embed_dim * 2, embed_dim // 2, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, q, k):
        r"""
        Args:
            q: cls features, shape (bs, num_frames, hid_dim)
            k: tokens features, shape (bs, num_tokens, hid_dim)
        """
        if self._qk_same_embed_dim is False:
            q = self.q_proj(q)
            k = self.k_proj(k)

        num_frames, num_tokens = q.size(1), k.size(1)
        # proj q
        q = q.transpose(-2, -1)
        q = self.qk_proj(q)   # q.repeat(1, num_tokens // num_frames, 1)
        q = q.transpose(-1, -2)
        qk = torch.cat([q, k], dim=-1)
        attn_weight = self.attn_proj(qk)
        attn_weight = attn_weight.squeeze(-1)
        attn_weight = attn_weight / attn_weight.sum(dim=1, keepdim=True)  # attn_weight = attn_weight.softmax(dim=1)
        return attn_weight


class AttentionXWeightPredictor(nn.Module):
    __constants__ = ['num_frames', 'num_tokens', 'embed_dim', 'num_heads', 'qk_scale', 'attn_drop', 'qdim', 'kdim']

    def __init__(self, num_frames: int, num_tokens: int, embed_dim: int, num_heads: int = 8, qk_bias: bool = False,
                 qk_scale: float = 1.0, attn_drop: float = 0., qdim: int = None, kdim: int = None, agg: str = "sum"):
        super().__init__()
        self.num_frames = num_frames
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = qk_scale
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self._qk_same_embed_dim = self.qdim == embed_dim and self.kdim == embed_dim
        self._agg_type = agg

        self.q_proj = nn.Linear(self.qdim, embed_dim, bias=qk_bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=qk_bias)
        self.attn_proj = nn.Linear(self.num_frames, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k):
        r"""
        Args:
            q: cls features, shape (bs, num_frames, hid_dim)
            k: tokens features, shape (bs, num_tokens, hid_dim)
        """

        if self._qk_same_embed_dim is False:
            q = self.q_proj(q)
            k = self.k_proj(k)
        num_frames, num_tokens = q.size(1), k.size(1)
        q = q.reshape(-1, num_frames, self.num_heads, self.head_dim)
        k = k.reshape(-1, num_tokens, self.num_heads, self.head_dim)
        assert q.size(0) == k.size(0)
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_frames, head_dim]
        k = k.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_tokens, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # [batch_size, num_heads, num_frames, max_words]
        attn = self.attn_drop(attn)
        if self._agg_type == "sum":
            attn_weight = attn.sum(dim=1)
        elif self._agg_type == "mean":
            attn_weight = attn.mean(dim=1)
        else:
            attn_weight, _ = torch.max(attn, dim=1)
        attn_weight = attn_weight.transpose(-2, -1)
        attn_weight = self.attn_proj(attn_weight)
        attn_weight = attn_weight.squeeze(-1).softmax(dim=1)
        return attn_weight

class TokenImportanceSelector(nn.Module):
    def __init__(self, thresh):
        super().__init__()
        thresh = thresh * torch.ones(1)
        self.register_buffer('thresh', thresh)

    def forward(self, x, attn_weight):
        batch_size, num_tokens, embed_dim = attn_weight.size(0), attn_weight.size(1), x.size(2)
        weights, weights_idx = attn_weight.sort(dim=1, descending=True)
        cumulative_importance = weights.cumsum(dim=1)
        # cumulative importance mask
        importance_mask = (cumulative_importance < self.thresh).to(attn_weight.dtype).reshape(-1, num_tokens)
        select_policy = torch.zeros(batch_size, num_tokens, dtype=importance_mask.dtype, device=importance_mask.device)
        select_policy = select_policy.scatter(dim=1, index=weights_idx, src=importance_mask)
        select_policy = torch.ones_like(select_policy, device=x.device) - select_policy
        select_condition = select_policy.unsqueeze(-1).bool().repeat(1, 1, embed_dim)
        # select according to importance mask
        x_zeros = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        y = torch.where(select_condition, x, x_zeros)
        return y, select_policy


if __name__ == "__main__":
    def test():
        cls = torch.ones([8, 32, 512])
        tokens = torch.randn([128, 48, 512])
        bs, num_frames, embed_dim = cls.size()
        bs, num_tokens, embed_dim = tokens.size()
        print("cls:\n   ", cls)
        print("tokens:\n    ", tokens)
        en_cls = cls.repeat(bs, 1, 1)  # [[t1], [t2]]->[[t1], [t2], [t1], [t2]]
        en_tokens = tokens.repeat_interleave(bs, 0)  # [[v1], [v2]]->[[v1], [v1], [v2], [v2]]
        print("en cls:\n   ", en_cls, "en cls shape:\n    ", en_cls.shape)
        print("en tokens:\n    ", en_tokens, "en tokens shape:\n    ", en_tokens.shape)
        # predictor = LinearXWeightPredictor(num_frames=num_frames, num_tokens=num_tokens, embed_dim=embed_dim)
        predictor = AttentionXWeightPredictor(num_frames=num_frames, num_tokens=num_tokens, embed_dim=embed_dim)
        attn_weights = predictor(en_cls, en_tokens)
        print("attn_weights:\n    ", attn_weights, "attn_weights shape:\n    ", attn_weights.shape)
        # global cls
        global_cls = torch.einsum('abd,ab->ad', en_tokens, attn_weights)
        print("global_cls:\n    ", global_cls, "global_cls shape:\n    ", global_cls.shape)
        cis = TokenImportanceSelector(0.6)
        masked_tokens, select_policy = cis(en_tokens, attn_weights)
        print("mask tokens:\n    ", masked_tokens, "mask tokens shape:\n    ", masked_tokens.shape)
        global_cls_partial = torch.einsum('abd,ab->ad', masked_tokens, attn_weights)
        print("global_cls_partial:\n    ", global_cls_partial,
              "global_cls_partial shape:\n    ", global_cls_partial.shape)
