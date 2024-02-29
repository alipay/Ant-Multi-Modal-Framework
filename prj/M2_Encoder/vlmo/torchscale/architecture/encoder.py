# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import numpy as np
import torch
import torch.nn as nn
from fairscale.nn import checkpoint_wrapper, wrap

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm

from vlmo.torchscale.architecture.utils import init_bert_params
from vlmo.torchscale.component.droppath import DropPath
from vlmo.torchscale.component.feedforward_network import FeedForwardNetwork, make_experts
from vlmo.torchscale.component.multihead_attention import MultiheadAttention
from vlmo.torchscale.component.multiway_network import MultiwayWrapper, set_split_position
from vlmo.torchscale.component.relative_position_bias import RelativePositionBias
from vlmo.torchscale.component.xmoe.moe_layer import MOELayer
from vlmo.torchscale.component.xmoe.routing import Top1Gate, Top2Gate
from vlmo.modules.vlmo_utils import no_sync_module_apply
from pytorch_lightning.utilities.distributed import rank_zero_info


class EncoderLayer(nn.Module):
    def __init__(self, args, depth, attn=None, is_moe_layer=False, is_encoder_decoder=False):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, args) if attn is None else attn
        self.self_attn_layer_norm = MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))
        self.dropout_module = torch.nn.Dropout(args.dropout)

        if args.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, args.drop_path_rate, args.encoder_layers)[depth]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.normalize_before = args.encoder_normalize_before
        self.is_moe_layer = is_moe_layer
        self.ffn_dim = args.encoder_ffn_embed_dim

        if not self.is_moe_layer:
            self.ffn = MultiwayWrapper(
                args,
                self.build_ffn(
                    self.embed_dim,
                    self.args,
                ),
            )
        else:
            assert not self.args.multiway
            if args.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    use_fp32=args.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    args.moe_gating_use_fp32,
                    args.moe_second_expert_policy,
                    args.moe_normalize_gate_prob_before_dropping,
                    args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            experts = make_experts(args, self.embed_dim, self.ffn_dim)
            self.moe_layer = MOELayer(gate, experts, args)
        self.final_layer_norm = MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))

        if args.deepnorm:
            if is_encoder_decoder:
                self.alpha = math.pow(math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625) * 0.81
            else:
                self.alpha = math.pow(2.0 * args.encoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def build_ffn(self, embed_dim, args):
        return FeedForwardNetwork(
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
            args.layernorm_eps,
            args.subln,
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
            one_attn=args.one_attn,
        )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        x,
        encoder_padding_mask,
        attn_mask=None,
        rel_pos=None,
        multiway_split_position=None,
        incremental_state=None,
    ):
        if multiway_split_position is not None:
            assert self.args.multiway
            no_sync_module_apply(self, set_split_position(multiway_split_position))

        if attn_mask is not None:
            # float16: -1e8 equal 0
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            rel_pos=rel_pos,
            incremental_state=incremental_state,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer:
            x = self.ffn(x)
            l_aux = None
        else:
            x = x.transpose(0, 1)
            x, l_aux = self.moe_layer(x)
            x = x.transpose(0, 1)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, l_aux


class Encoder(nn.Module):
    def __init__(
        self, args, embed_tokens=None, embed_positions=None, output_projection=None, is_encoder_decoder=False, **kwargs
    ):
        self.args = args
        super().__init__(**kwargs)

        self.dropout_module = torch.nn.Dropout(args.dropout)

        embed_dim = args.encoder_embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)
        self.mask_ratio = args.mask_ratio
        self.max_text_len = args.max_text_len
        self.vision_len = (args.img_size // args.patch_size) * (args.img_size // args.patch_size)

        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        if output_projection is None and not is_encoder_decoder and not args.no_output_layer and args.vocab_size > 0:
            self.output_projection = self.build_output_projection(args)
        else:
            self.output_projection = output_projection

        if args.layernorm_embedding:
            self.layernorm_embedding = MultiwayWrapper(args, LayerNorm(embed_dim, eps=args.layernorm_eps), dim=1)
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])
        if self.args.share_layer:
            single_layer = self.build_encoder_layer(
                args, depth=0, is_moe_layer=False, is_encoder_decoder=is_encoder_decoder
            )
            for i in range(args.encoder_layers):
                self.layers.append(single_layer)
        elif self.args.share_attn:
            moe_freq = args.moe_freq
            embed_dim = args.encoder_embed_dim
            shared_attn = self.build_self_attention(embed_dim, self.args)
            for i in range(args.encoder_layers):
                is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
                self.layers.append(
                    self.build_encoder_layer(
                        args,
                        depth=i,
                        attn=shared_attn,
                        is_moe_layer=is_moe_layer,
                        is_encoder_decoder=is_encoder_decoder,
                    )
                )

        else:
            moe_freq = args.moe_freq
            for i in range(args.encoder_layers):
                is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
                self.layers.append(
                    self.build_encoder_layer(
                        args,
                        depth=i,
                        is_moe_layer=is_moe_layer,
                        is_encoder_decoder=is_encoder_decoder,
                    )
                )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before and args.normalize_output:
            self.layer_norm = MultiwayWrapper(args, LayerNorm(embed_dim, eps=args.layernorm_eps))
        else:
            self.layer_norm = None

        if args.rel_pos_buckets > 0 and args.max_rel_pos > 0:
            self.relative_position = RelativePositionBias(
                num_buckets=args.rel_pos_buckets,
                max_distance=args.max_rel_pos,
                n_heads=args.encoder_attention_heads,
            )
        else:
            self.relative_position = None

        if args.bert_init:
            self.apply(init_bert_params)

        if args.deepnorm:
            if is_encoder_decoder:
                init_scale = math.pow(math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625) / 1.15
            else:
                init_scale = math.pow(8.0 * args.encoder_layers, 0.25)
            for name, p in self.named_parameters():
                if "fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name:
                    p.data.div_(init_scale)

        if args.subln:
            if is_encoder_decoder:
                init_scale = math.sqrt(math.log(3 * args.decoder_layers) * math.log(2 * args.encoder_layers) / 3)
            else:
                init_scale = math.sqrt(math.log(args.encoder_layers * 2))
            for name, p in self.named_parameters():
                if "fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name:
                    p.data.mul_(init_scale)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L - 1, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1) + torch.ones(N, L - 1, device=x.device, dtype=int)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        x0 = x[:, 0, :]
        x0 = x0.reshape(N, 1, D)
        x_masked_add = torch.cat([x0, x_masked], axis=1)
        return x_masked_add, ids_keep

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
            one_attn=args.one_attn,
        )

    def build_output_projection(
        self,
        args,
    ):
        if args.share_encoder_input_output_embed:
            assert args.encoder_embedding_type == "language"
            output_projection = torch.nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(args.encoder_embed_dim, args.vocab_size, bias=False)
            torch.nn.init.normal_(output_projection.weight, mean=0, std=args.encoder_embed_dim**-0.5)
        return output_projection

    def checkpointing_and_params_allgather(
        self,
        origin_layer,
    ):
        origin_forward = origin_layer.forward

        from deepspeed import checkpointing
        def forward(*args, **kwargs):
            # deepspeed checkpoint not support kwargs
            ret = checkpointing.checkpoint(origin_forward, *args, **kwargs)
            return ret

        return forward

    def build_encoder_layer(self, args, depth, attn=None, is_moe_layer=False, is_encoder_decoder=False):
        layer = EncoderLayer(
            args,
            depth,
            attn,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            rank_zero_info("EncoderLayer params: %s", sum(p.numel() for p in layer.parameters() if p.requires_grad))
            layer = checkpoint_wrapper(layer)
            # layer.ffn = checkpoint_wrapper(layer.ffn,)
        if args.fsdp:
            layer = wrap(layer)
        return layer

    def checkpointing_layers(self):
        for i, layer in enumerate(self.layers):
            rank_zero_info(f"Checkpointing wrapper EncoderLayers: {i}")
            self.layers[i] = checkpoint_wrapper(layer)

    def forward_embedding(
        self,
        src_tokens,
        token_embedding=None,
        positions=None,
    ):
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            if src_tokens is not None:
                x = embed + self.embed_positions(src_tokens, positions=positions)
            else:
                x = embed + self.embed_positions(x, positions=positions)
        is_flip, ids_keep = 0, None
        if self.mask_ratio > 0:
            if x.shape[1] == self.vision_len + 1:
                x, ids_keep = self.random_masking(x, self.mask_ratio)
                is_flip = 1
            elif x.shape[1] == self.vision_len + self.max_text_len + 1:
                vision_tokens = x[:, : self.vision_len + 1, :]
                vision_tokens, ids_keep = self.random_masking(vision_tokens, self.mask_ratio)
                x = torch.cat(
                    [
                        vision_tokens,
                        x[
                            :,
                            self.vision_len + 1 :,
                        ],
                    ],
                    dim=1,
                )
                is_flip = 2
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed, ids_keep, is_flip

    def forward(
        self,
        src_tokens,
        encoder_padding_mask=None,
        attn_mask=None,
        return_all_hiddens=False,
        token_embeddings=None,
        multiway_split_position=None,
        features_only=False,
        incremental_state=None,
        positions=None,
        **kwargs
    ):
        assert src_tokens is not None or token_embeddings is not None

        if encoder_padding_mask is None:
            if src_tokens is not None:
                encoder_padding_mask = torch.zeros_like(src_tokens, device=src_tokens.device).bool()
            else:
                encoder_padding_mask = torch.zeros(
                    [token_embeddings.size(0), token_embeddings.size(1)],
                    device=token_embeddings.device,
                ).bool()

        if multiway_split_position is not None:
            assert self.args.multiway
            no_sync_module_apply(self, set_split_position(multiway_split_position))

        x, encoder_embedding, ids_keep, is_flip = self.forward_embedding(src_tokens, token_embeddings, positions)
        if is_flip > 0:
            if is_flip == 2:
                text_ids = (
                    torch.arange(
                        self.vision_len + 1, self.vision_len + 1 + self.max_text_len, device=x.device, dtype=torch.int64
                    )
                    .unsqueeze(0)
                    .repeat(ids_keep.shape[0], 1)
                )
                cls_ids = torch.zeros(ids_keep.shape[0], 1, device=x.device, dtype=torch.int64)
                ids_keep = torch.cat([cls_ids, ids_keep, text_ids], dim=1)
            elif is_flip == 1:
                cls_ids = torch.zeros(ids_keep.shape[0], 1, device=x.device, dtype=torch.int64)
                ids_keep = torch.cat([cls_ids, ids_keep], dim=1)
            if encoder_padding_mask is not None:
                encoder_padding_mask = torch.gather(encoder_padding_mask, dim=1, index=ids_keep)
            if attn_mask is not None:
                attn_mask = torch.gather(
                    attn_mask, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, attn_mask.shape[-1])
                )
                attn_mask = torch.gather(attn_mask, dim=2, index=ids_keep.unsqueeze(1).repeat(1, attn_mask.shape[1], 1))
            if multiway_split_position > 0:
                multiway_split_position = ids_keep.shape[1] - self.max_text_len
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        rel_pos_bias = None
        if self.relative_position is not None:
            rel_pos_bias = self.relative_position(batch_size=x.size(0), qlen=x.size(1), klen=x.size(1))

        l_aux = []
        for idx, layer in enumerate(self.layers):
            x, l_aux_i = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if incremental_state is None else None,
                attn_mask=attn_mask,
                rel_pos=rel_pos_bias,
                multiway_split_position=multiway_split_position,
                incremental_state=incremental_state[idx] if incremental_state is not None else None,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            l_aux.append(l_aux_i)

        if multiway_split_position is not None:
            assert self.args.multiway
            no_sync_module_apply(self, set_split_position(multiway_split_position))
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not features_only and self.output_projection is not None:
            x = self.output_projection(x)

        return {
            "encoder_out": x,
            "encoder_embedding": encoder_embedding,
            "encoder_padding_mask": encoder_padding_mask,
            "encoder_states": encoder_states,
            "l_aux": l_aux,
            "multiway_split_position": multiway_split_position,
        }
