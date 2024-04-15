import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
from typing import Dict, List, Callable, Any

import transformers
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, LlamaConfig, LlamaMLP, LlamaRMSNorm, LlamaDecoderLayer, LlamaModel, LlamaFlashAttention2, LlamaSdpaAttention
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPast
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPConfig, CLIPEncoderLayer, CLIPMLP
from .eva_vit import Attention as EvaAttention

import pink.model.eva_vit

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch.cuda.amp import autocast


class AdapterLayer(nn.Module):
    def __init__(
        self, 
        in_features,
        hidden_dim=8, 
        scale=1,
        dropout=0.1,
        non_linear=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.in_features = in_features
        self.tune_adapter_a = nn.Linear(self.in_features, hidden_dim, bias=True)
        self.tune_adapter_b = nn.Linear(hidden_dim, self.in_features, bias=True)
        self.dropout = nn.Dropout(dropout)

        if non_linear:
            self.activate = nn.SiLU()
        else:
            self.activate = nn.Identity()

    def train(self, mode: bool = True):
        self.tune_adapter_a.train(mode)
        self.tune_adapter_b.train(mode)
        self.dropout.train(mode)

    def forward(self, x):
        previous_dtype = x.dtype
        weight_dtype = self.tune_adapter_a.weight.data.dtype
        down_x = self.tune_adapter_a(x.to(weight_dtype))
        down_x = self.activate(down_x)
        up_x = self.tune_adapter_b(self.dropout(down_x))
        result = up_x.to(previous_dtype) + x
        return result


def mark_only_adapter_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias: 
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if 'adapter_' not in n:
            p.requires_grad = False
        else:
            p.data = p.data.float()
            p.requires_grad = True

    # depending on the `bias` value unfreeze bias weights
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    else:
        raise NotImplementedError

@dataclass
class AdapterConfig:
    hidden_dim: int = 8
    scale: float = 1.0
    dropout: float = 0.1
    adapter_attn: bool = True
    adapter_mlp: bool = True
    non_linear: bool = False


class CLIPEncoderAdapterLayer(nn.Module):
    adapter_config = None
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.adapter_config.adapter_attn:
            self.adapter_attn = AdapterLayer(
                self.embed_dim,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
                self.adapter_config.non_linear,
            )
        if self.adapter_config.adapter_mlp:
            self.adapter_mlp = AdapterLayer(
                self.embed_dim,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
                self.adapter_config.non_linear,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        if hasattr(self, "adapter_attn"):
            assert self.adapter_config.adapter_attn
            hidden_states = self.adapter_attn(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        if hasattr(self, "adapter_mlp"):
            assert self.adapter_config.adapter_mlp
            hidden_states = self.adapter_mlp(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class EvaAdapterAttention(nn.Module):
    adapter_config = None
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.adapter_config.adapter_attn:
            self.adapter_attn = AdapterLayer(
                self.dim,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
                self.adapter_config.non_linear,
            )

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        if hasattr(self, "adapter_attn"):
            assert self.adapter_config.adapter_attn
            x = self.adapter_attn(x)
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaAdapterDecoderLayer(nn.Module):
    adapter_config = None
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.adapter_config.adapter_attn:
            self.adapter_attn = AdapterLayer(
                self.hidden_size,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
                self.adapter_config.non_linear,
            )
        if self.adapter_config.adapter_mlp:
            self.adapter_mlp = AdapterLayer(
                self.hidden_size,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if self.adapter_config.adapter_attn:
            hidden_states = self.adapter_attn(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.adapter_config.adapter_mlp:
            hidden_states = self.adapter_mlp(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@contextmanager
def adapter(hidden_dim, scale, dropout, enabled: bool = True, non_linear=False, attn=True, mlp=False):
    if not enabled:
        yield
        return

    LlamaAdapterDecoderLayer.adapter_config = AdapterConfig(hidden_dim=hidden_dim, scale=scale, dropout=dropout, non_linear=non_linear, adapter_attn=attn, adapter_mlp=mlp)
    original_layer = LlamaDecoderLayer
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = LlamaAdapterDecoderLayer
    yield
    # when exiting context manager - restore link to original causal self-attention class
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = original_layer


@contextmanager
def visual_adapter(hidden_dim, scale, dropout, attn=True, mlp=False, enabled: bool = True, non_linear=False):
    if not enabled:
        yield
        return

    CLIPEncoderAdapterLayer.adapter_config = AdapterConfig(hidden_dim=hidden_dim, scale=scale, dropout=dropout, adapter_attn=attn, adapter_mlp=mlp, non_linear=False)
    original_layer = CLIPEncoderLayer
    transformers.models.clip.modeling_clip.CLIPEncoderLayer = CLIPEncoderAdapterLayer
    yield
    transformers.models.clip.modeling_clip.CLIPEncoderLayer = original_layer


@contextmanager
def eva_adapter(hidden_dim, scale, dropout, attn=True, mlp=False, enabled: bool = True, non_linear=False):
    if not enabled:
        yield
        return

    EvaAdapterAttention.adapter_config = AdapterConfig(hidden_dim=hidden_dim, scale=scale, dropout=dropout, adapter_attn=attn, adapter_mlp=mlp, non_linear=False)
    original_layer = EvaAttention
    pink.model.eva_vit.Attention = EvaAdapterAttention
    yield
    pink.model.eva_vit.Attention = original_layer
