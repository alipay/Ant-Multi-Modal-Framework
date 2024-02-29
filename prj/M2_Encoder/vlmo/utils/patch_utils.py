# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def _patch_forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
):
    bsz, tgt_len, embed_dim = query.size()
    src_len = tgt_len
    assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

    key_bsz, src_len, _ = key.size()
    assert key_bsz == bsz, f"{query.size(), key.size()}"
    assert value is not None
    assert bsz, src_len == value.shape[:2]

    q = self.q_proj(query)
    k = self.k_proj(key)
    v = self.v_proj(value)

    q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

    if incremental_state is not None or self.xpos is not None:
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)
        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(bsz * self.num_heads, -1, self.head_dim)
                prev_value = incremental_state["prev_value"].view(bsz * self.num_heads, -1, self.head_dim)
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            incremental_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)
        q = q.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz, self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz, self.num_heads, src_len, self.head_dim)

    assert rel_pos is None

    # move repeat_interleave to encoder.py is useless?(recompute will save more tensor)
    if attn_mask is not None:
        if len(attn_mask.shape) == 2:
            attn_mask = attn_mask.unsqueeze(0).repeat_interleave(bsz * self.num_heads, dim=0)
        else:
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)

    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        key_padding_mask = key_padding_mask.repeat_interleave(tgt_len, dim=2)
        key_padding_mask = key_padding_mask.repeat_interleave(self.num_heads, dim=1)
        key_padding_mask = key_padding_mask.view(bsz * self.num_heads, tgt_len, src_len)
        if attn_mask is not None:
            attn_mask.masked_fill_(key_padding_mask.to(torch.bool), -torch.inf)
        else:
            attn_mask = key_padding_mask.to(q.dtype).masked_fill(key_padding_mask.to(torch.bool), -torch.inf)
    if attn_mask is not None:
        attn_mask = attn_mask.to(q.dtype).reshape(bsz, self.num_heads, *tuple(attn_mask.shape[-2:]))
    with torch.backends.cuda.sdp_kernel(enable_math=False if attn_mask is None else True):
        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_module.p if self.training else 0.0,
        )

    attn = attn.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)

    if self.inner_attn_ln is not None:
        attn = self.inner_attn_ln(attn)

    attn = self.out_proj(attn)
    # encoder未使用attn weight，直接返回None
    return attn, None


def patch_torch_scale_with_flash_attn():
    from vlmo.torchscale.component.multihead_attention import MultiheadAttention
    torch.backends.cuda.enable_flash_sdp(True)
    MultiheadAttention._origin_forward = MultiheadAttention.forward
    MultiheadAttention.forward = _patch_forward
    print('Finish patch_torch_scale_with_flash_attn!')
