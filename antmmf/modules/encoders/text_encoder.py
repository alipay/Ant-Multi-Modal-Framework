# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import math
import os.path as osp
import warnings

import torch
from torch import nn
from torch.nn import TransformerEncoder as Enc
from torch.nn import TransformerEncoderLayer as Layer
from transformers import AutoConfig, AutoModel

from antmmf.common import Configuration, configurable
from antmmf.modules.embeddings import TextEmbedding
from antmmf.modules.module_registry import ModuleRegistry
from antmmf.modules.utils import get_mask
from antmmf.modules.vision.backbone.clip.model import CLIPLanguageEncoder
from antmmf.utils.general import get_transformer_model_vocab_path


class TextEncoder(ModuleRegistry):
    def __init__(self, config: Configuration, *args, **kwargs):
        config_kwargs = config.get("params", {})
        super(TextEncoder, self).__init__(config.type, *args, **config_kwargs, **kwargs)


TextEncoder.register(nn.Identity)
TextEncoder.register(CLIPLanguageEncoder)


class PretrainedTransformerEncoderAndEmbedding(nn.Module):
    """
    support loading offline bert model, ${PYTORCH_TRANSFORMERS_CACHE} indicates absolute path to bert_model_dir
    bert_model_dir should have this hier structure:
    |____bert_model_dir
    | |____bert-base-uncased # bert_model_name
    | | |____config.json
    | | |____pytorch_model.bin
    | | |____vocab.txt

    Args:

        num_segments (int):
        model_type (str): determine which model config to load offline, instead of downloading config.json online.
         model_types are defined in:
         https://github.com/huggingface/transformers/blob/master/src/transformers/models/auto/configuration_auto.py#L25-L91
        bert_model_name (str):
        hidden_size (int):
        intermediate_size (int):
        num_hidden_layers (int):
        start_hidden_layer (int):
        num_attention_heads (int):
        output_attentions (bool):
        output_hidden_states (bool):
        vocab_size (int): modify this based on bert_model_name
        gradient_checkpointing (bool):
        type_vocab_size (int):
        max_position_embeddings (int)
    """

    @configurable
    def __init__(
        self,
        pretrained: bool = None,
        num_segments: int = None,
        model_type: str = "bert",
        bert_model_name: str = "bert-base-uncased",
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
    ):
        super().__init__()

        config = Configuration(locals())
        del config["self"]
        del config["__class__"]

        # load complete pretrained model first, then tailed to our needs
        hf_params = {"config": AutoConfig.for_model(**config)}

        if pretrained is False:
            config = AutoConfig.for_model(**hf_params["config"].to_dict())
            self.module = AutoModel.from_config(config)
            warnings.warn("random initialization for {}".format(bert_model_name))
        else:
            pretrained_model_path = get_transformer_model_vocab_path(bert_model_name)
            if not osp.exists(pretrained_model_path):
                # roll back to model_type to trigger auto-downloading
                pretrained_model_path = osp.basename(pretrained_model_path)
            self.module = AutoModel.from_pretrained(pretrained_model_path, **hf_params)

        # tail BERT layers we do not need
        self.module.encoder.layer = nn.ModuleList(
            iter(
                self.module.encoder.layer[
                    start_hidden_layer : start_hidden_layer + num_hidden_layers
                ]
            )
        )

        self.embeddings = None
        for att in ["embeddings", "word_embedding"]:
            if hasattr(self.module, att):
                self.embeddings = getattr(self.module, att)

        self.num_segments = num_segments
        self.config = self.module.config
        self._init_segment_embeddings()

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

    def forward(self, *args, return_sequence=False, **kwargs):
        output = self.module(*args, **kwargs)
        return output[0] if return_sequence else output[1]


@TextEncoder.register()
class PretrainedTransformerEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        pretrained: bool = None,
        num_segments: int = None,
        model_type: str = "bert",
        bert_model_name: str = "bert-base-uncased",
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
    ):
        super(PretrainedTransformerEncoder, self).__init__()
        config = Configuration(locals())
        del config["self"]
        del config["__class__"]
        module = PretrainedTransformerEncoderAndEmbedding(**config).module
        self.encoder = module.encoder
        self.embeddings = module.embeddings
        self.pooler = module.pooler
        self.module = module

    def forward(self, *args, return_dict=False, **kwargs):
        return self.module(*args, **kwargs, return_dict=return_dict)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).squeeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        bsz, length, dim = x.size()
        x = x + self.pe[:length, :]
        return self.dropout(x)


@TextEncoder.register()
class TextTransformerEncoderModel(nn.Module):
    @configurable
    def __init__(
        self, ninp: int, nhead: int, nhid: int, nlayers: int, dropout: float = 0.5
    ):
        super(TextTransformerEncoderModel, self).__init__()

        self.model_type = "TransformerEncoderModel"
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = Layer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = Enc(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, src_length=None, src_key_padding_mask=None):
        # binary mask of valid source vs padding
        bsz, length, _ = src.size()
        src_key_padding_mask = (
            ((1.0 - get_mask(src_length, length)).type(torch.bool).to(src.device))
            if src_key_padding_mask is None
            else src_key_padding_mask
        )

        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        src = src.transpose(0, 1)
        output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )
        output = output.transpose(0, 1)

        return output, src_key_padding_mask


@TextEncoder.register()
class TextEmbeddingEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        operator: str,
        embedding_params: Configuration,
    ):
        super().__init__()
        self._operator = operator
        self._embedding_params = embedding_params
        if self._operator == "pretrained":
            self.module = PretrainedTransformerEncoderAndEmbedding(
                embedding_params
            ).embeddings
        else:
            self.module = TextEmbedding(
                self._embedding_params.type, **self._embedding_params.params
            )

    def forward(self, x):
        x = self.module(x)
        if self._operator == "sum":
            x = x.sum(dim=1)
        elif self._operator == "concat":
            x = torch.cat(x, dim=1)
        elif self._operator == "mul":
            x = torch.prod(x, dim=1)

        return x.squeeze()
