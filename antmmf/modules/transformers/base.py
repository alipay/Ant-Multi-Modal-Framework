# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

"""
custom transformer modules, which gives more control for transformer based models.
"""
import copy
from typing import Any, Dict, Union
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from dataclasses import field
from torch import nn, Tensor

from antmmf.common import Configuration, configurable


class Transformer(nn.Module):
    """
    DETR Transformer class.

    Copy-paste from torch.nn.Transformer with modifications:
        * positional encodings are passed in MHattention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        layer_params = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )

        self.encoder = TransformerEncoder(
            type="PositionEnhancedEncoderLayer",
            params=layer_params,
            num_layers=num_encoder_layers,
        )

        self.decoder = TransformerDecoder(
            type="PositionEnhancedDecoderLayer",
            params=layer_params,
            num_layers=num_decoder_layers,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        :param src(torch.float32): b,c,h,w
        :param mask(torch.bool): b, h, w
        :param query_embed(torch.float32): num_queries, hidden_dim
        :param pos_embed(torch.float32): b, c, h, w
        :return:
            memory: Encoder output, [bs, c, h, w]
            hs: Decoder output, [bs, c, h, w]
        """
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # b, c,  -> seq_length, b, c
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # seq_length, b, c
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # num_queries, b, c
        mask = mask.flatten(1)  # b, h, w -> b, seq_length

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # parallel decoding: https://github.com/facebookresearch/detr/issues/3
        # Note: No target mask is used.
        # For generation: https://github.com/pytorch/tutorials/issues/719#issuecomment-798983859
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    """
    Args:
        type (str):
        num_layers (int): the number of encoder layers
        params (dict): encoder layer config
        use_checkpoint (bool): Whether to use checkpointing to save memory. To use pytorch gradient_checkpointing,
         you must:
          1) training_parameters.find_unused_parameters False. Reference:
           https://discuss.pytorch.org/t/how-to-use-torch-nn-parallel-distributeddataparallel-and-torch-utils-checkpoint-together/96338/4
          2) Remove all params that do not have gradients(upgrade pytorch to 1.9 may be helpful). Reference:
           https://stackoverflow.com/questions/68000761/pytorch-ddp-finding-the-cause-of-expected-to-mark-a-variable-ready-only-once
          3) Do not use gradient_checkpointing for modules that share params, especially for transformer decoders.
    """

    @configurable
    def __init__(
        self,
        type: str,
        num_layers: int,
        params: Union[Dict[str, Any], Configuration] = field(default_factory=dict),
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()

        encoder_layer = TansformerEncoderFactory(
            Configuration(locals()), **kwargs
        ).module
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = (
            nn.LayerNorm(params["d_model"]) if params["normalize_before"] else None
        )
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                output = checkpoint.checkpoint(
                    layer, output, mask, src_key_padding_mask, pos
                )
            else:
                output = layer(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos=pos,
                )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """TransformerDecoder

    Args:
        type (str):
        num_layers (int): the number of decoder layers
        params (dict): decoder layer config
        use_checkpoint (bool): Whether to use checkpointing to save memory
    """

    @configurable
    def __init__(
        self,
        type: str,  # decoder layer
        num_layers: int,
        params: Dict[str, Any] = field(default_factory=dict),  # decoder layer config
        return_intermediate: bool = False,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        print(params)
        decoder_layer = TansformerDecoderFactory(
            Configuration(locals()), **kwargs
        ).module
        self.layers = _get_clones(decoder_layer, num_layers)

        self.norm = nn.LayerNorm(params["d_model"])
        self.return_intermediate = return_intermediate or params.get(
            "return_intermediate", False
        )
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    memory,
                    tgt_mask,
                    memory_mask,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                    pos,
                    query_pos,
                )
            else:
                output = layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    pos=pos,
                    query_pos=query_pos,
                )

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


# TansformerEncoder ===>
class TansformerEncoderFactory(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self._type = config.type
        if self._type == "PositionEnhancedEncoderLayer":
            self.module = PositionEnhancedEncoderLayer(config.params, **kwargs)
        else:
            raise NotImplementedError(f"Unknown Multi-modal Encoder {self._type}")

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class PositionEnhancedEncoderLayer(nn.Module):
    @configurable
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # see detail at: On Layer Normalization in the Transformer Architecture
        # https://openreview.net/forum?id=B1x8anVFPr
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# TansformerEncoder ===>

# TansformerDecoder ===>
class TansformerDecoderFactory(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self._type = config.type
        if self._type == "PositionEnhancedDecoderLayer":
            self.module = PositionEnhancedDecoderLayer(config.params, **kwargs)
        else:
            raise NotImplementedError(f"Unknown Multi-modal Encoder {self._type}")

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class PositionEnhancedDecoderLayer(nn.Module):
    @configurable
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = True,
    ):
        super().__init__()
        # see differences between *mask & *key_padding_mask:
        # https://zhuanlan.zhihu.com/p/353365423
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # see detail at: On Layer Normalization in the Transformer Architecture
        # https://openreview.net/forum?id=B1x8anVFPr
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


# TansformerDecoder <===


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
