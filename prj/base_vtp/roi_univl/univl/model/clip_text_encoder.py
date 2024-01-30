# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import os
import torch
from torch import nn

from antmmf.modules.encoders import TextEncoder
from antmmf.common import Configuration, configurable
from antmmf.modules.vision.backbone.clip.configuration_bert import BertConfig
from antmmf.modules.vision.backbone.clip.modeling_bert import BertModel, BertLayerNorm
from antmmf.modules.vision.backbone.clip.cn_model import _MODELS as CNMODELS
from antmmf.modules.vision.backbone.clip.cn_model import _download


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, input_ids=None, inputs_embeds=None, token_type_ids=None, position_ids=None
    ):
        if input_ids is None:
            input_ids = inputs_embeds[:, :, 0]

        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if inputs_embeds is not None:
            words_embeddings = inputs_embeds
        else:
            words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel2(BertModel):
    def __init__(self, config):
        super(BertModel2, self).__init__(config)
        self.embeddings = BertEmbeddings(config)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        encoder_outputs = self.encoder(
            embedding_output, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        return sequence_output, sequence_output[:, 0, :]


@TextEncoder.register()
class RobertBertEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: bool = True,
        num_segments: int = None,
        model_type: str = "bert",
        bert_model_name: str = "roberta_chinese_base",
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        start_hidden_layer: int = 0,
        num_attention_heads: int = 12,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        vocab_size: int = 30522,
        gradient_checkpointing: bool = False,
        type_vocab_size: int = 2,
        max_position_embeddings: int = 512,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-6,
        is_proj: bool = True,
        out_dim: 768 = int,
    ):
        super().__init__()
        config = Configuration(locals())
        del config["self"]
        del config["__class__"]
        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=1e-12,
        )
        module = BertModel2(
            self.bert_config
        )  # (**config) #PretrainedTransformerEncoderAndEmbedding(**config).module #BertModel(**config).module
        self.encoder = module.encoder
        self.embeddings = module.embeddings
        self.module = module
        self.out_dim = out_dim
        self.num_segments = num_segments
        self._init_segment_embeddings()
        if is_proj:
            self.text_projection = nn.Parameter(torch.empty(hidden_size, out_dim))
        else:
            self.text_projection = None
        if pretrained:
            self.load_state_dict(model_name)

    def load_state_dict(self, name, download_root=None):
        download_root = download_root or os.path.expanduser(os.environ["TORCH_HOME"])
        if name in CNMODELS:
            model_path = _download(CNMODELS[name], download_root)
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(
                f"Model {name} not found; available models = {available_models()}"
            )

        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        new_state_dict = {}
        for k in state_dict.keys():
            if "text_projection" not in k and "bert" not in k:
                continue
            if "module" in k and "bert" in k:
                new_state_dict[k[12:]] = state_dict[k]
            elif "module" in k:
                new_state_dict[k[7:]] = state_dict[k]
            else:
                new_state_dict[k[5:]] = state_dict[k]
        if new_state_dict is not None:
            for k in self.module.state_dict():
                if "module" in k:
                    self.module.state_dict()[k[7:]].copy_(new_state_dict[k[7:]])
                else:
                    self.module.state_dict()[k].copy_(new_state_dict[k])
        if self.text_projection is not None:
            if (
                self.text_projection.data.shape
                == new_state_dict["text_projection"].shape
            ):
                self.text_projection.data.copy_(new_state_dict["text_projection"])

    def _init_segment_embeddings(self):
        if self.num_segments is not None:
            if hasattr(self.embeddings, "token_type_embeddings"):
                # avoid copying when num_segments not change
                if (
                    self.num_segments
                    == self.embeddings.token_type_embeddings.num_embeddings
                ):
                    return
                new_embeds = nn.Embedding(self.num_segments, self.config.hidden_size)
                new_embeds.weight.data[:2].copy_(
                    self.embeddings.token_type_embeddings.weight
                )
                for idx in range(2, self.num_segments - 1):
                    new_embeds.weight.data[idx].copy_(
                        self.embeddings.token_type_embeddings.weight.data.mean(dim=0)
                    )
                self.embeddings.token_type_embeddings = new_embeds

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        output_attentions=False,
    ):
        sequence_output, pooled_output = self.module(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask
        )
        if self.text_projection is not None:
            return sequence_output, sequence_output[:, 0, :] @ self.text_projection
        else:
            return sequence_output, sequence_output[:, 0, :]
