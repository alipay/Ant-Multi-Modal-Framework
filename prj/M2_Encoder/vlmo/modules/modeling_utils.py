# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from vlmo.torchscale.model.BEiT3 import BEiT3
from vlmo.torchscale.architecture.config import EncoderConfig


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_base_config(
    img_size=224,
    patch_size=16,
    drop_path_rate=0,
    checkpoint_activations=None,
    mlp_ratio=4,
    vocab_size=64010,
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_attention_heads=12,
    share_layer=False,
    share_attn=False,
    deepnorm=False,
    mask_ratio=0,
    max_text_len=52,
    one_attn=False,
    **kwargs
):
    return EncoderConfig(
        img_size=img_size,
        patch_size=patch_size,
        vocab_size=vocab_size,
        multiway=True,
        layernorm_embedding=False,
        normalize_output=True,
        no_output_layer=True,
        drop_path_rate=drop_path_rate,
        encoder_embed_dim=encoder_embed_dim,
        encoder_attention_heads=encoder_attention_heads,
        encoder_layers=encoder_layers,
        encoder_ffn_embed_dim=int(encoder_embed_dim * mlp_ratio),
        checkpoint_activations=checkpoint_activations,
        share_layer=share_layer,
        share_attn=share_attn,
        deepnorm=deepnorm,
        mask_ratio=mask_ratio,
        max_text_len=max_text_len,
        one_attn=one_attn,
    )


def _get_large_config(
    img_size=224,
    patch_size=16,
    drop_path_rate=0,
    checkpoint_activations=None,
    mlp_ratio=4,
    vocab_size=64010,
    encoder_layers=24,
    encoder_embed_dim=1024,
    encoder_attention_heads=16,
    share_layer=False,
    share_attn=False,
    deepnorm=False,
    mask_ratio=0,
    max_text_len=52,
    one_attn=False,
    **kwargs
):
    return EncoderConfig(
        img_size=img_size,
        patch_size=patch_size,
        vocab_size=vocab_size,
        multiway=True,
        layernorm_embedding=False,
        normalize_output=True,
        no_output_layer=True,
        drop_path_rate=drop_path_rate,
        encoder_embed_dim=encoder_embed_dim,
        encoder_attention_heads=encoder_attention_heads,
        encoder_layers=encoder_layers,
        encoder_ffn_embed_dim=int(encoder_embed_dim * mlp_ratio),
        checkpoint_activations=checkpoint_activations,
        share_layer=share_layer,
        share_attn=share_attn,
        deepnorm=deepnorm,
        mask_ratio=mask_ratio,
        max_text_len=max_text_len,
        one_attn=one_attn,
    )


def _get_huge_config(
    img_size=224,
    patch_size=16,
    drop_path_rate=0,
    checkpoint_activations=None,
    mlp_ratio=4,
    vocab_size=30522,
    encoder_layers=32,
    encoder_embed_dim=4096,
    encoder_attention_heads=32,
    share_layer=False,
    share_attn=False,
    deepnorm=False,
    mask_ratio=0,
    max_text_len=52,
    one_attn=False,
    **kwargs
):
    return EncoderConfig(
        img_size=img_size,
        patch_size=patch_size,
        vocab_size=vocab_size,
        multiway=True,
        layernorm_embedding=False,
        normalize_output=True,
        no_output_layer=True,
        drop_path_rate=drop_path_rate,
        encoder_embed_dim=encoder_embed_dim,
        encoder_attention_heads=encoder_attention_heads,
        encoder_layers=encoder_layers,
        encoder_ffn_embed_dim=int(encoder_embed_dim * mlp_ratio),
        checkpoint_activations=checkpoint_activations,
        share_layer=share_layer,
        share_attn=share_attn,
        deepnorm=deepnorm,
        mask_ratio=mask_ratio,
        max_text_len=max_text_len,
        one_attn=one_attn,
    )


class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed",
            "cls_token",
            "beit3.encoder.embed_positions.A.weight",
            "beit3.vision_embed.cls_token",
            "logit_scale",
        }

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
