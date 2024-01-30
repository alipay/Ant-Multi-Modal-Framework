# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import warnings
import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from antmmf.utils.general import get_transformer_model_vocab_path
from antmmf.common import constants, configurable, Configuration
from antmmf.common.registry import registry
from antmmf.modules.embeddings import ImageBertEmbeddings
from antmmf.modules.module_registry import ModuleRegistry
from antmmf.modules.transformers.base import TransformerEncoder
from .visual_encoder import VisualEncoder


class MultimodalEncoder(ModuleRegistry):
    def __init__(self, config: Configuration, **kwargs):
        encoder_type = config.type
        super(MultimodalEncoder, self).__init__(encoder_type, config.params, **kwargs)


MultimodalEncoder.register(nn.Identity)
MultimodalEncoder.register(TransformerEncoder)


@MultimodalEncoder.register()
class MultimodalBertEncoder(nn.Module):
    r"""
    This is the class that use AutoModel to encode multi-modality
    """

    @configurable
    def __init__(
        self,
        image_encoder: Configuration,
        bert_model_name: str,
        model_type: str = "bert",
        **kwargs,
    ):
        r"""
        Instantiation of this class from a configuration

        Args:
            config is a dictionary providing options below:

            bert_model_name: the Bert model name from AutoModel
            pretrained: false if requesting to initialize the model weights from scratch
                        otherwise using pretrained weights. Note that, by default,
                        this value is true, without specification.
        """
        super(MultimodalBertEncoder, self).__init__()
        self.image_encoder = image_encoder

        self.writer = registry.get("writer")

        self.output_rep_info = False
        if kwargs.get(constants.OUTPUT_REPRESENTATIONS):
            self.output_rep_info = True

        if kwargs.get(constants.PRETRAINED_STR) is False:
            bert_config = AutoConfig.for_model(
                model_type=model_type, bert_model_name=bert_model_name, **kwargs
            )
            bert = AutoModel.from_config(bert_config)
            warnings.warn("random initialization for {}".format(bert_model_name))
        else:
            pretrained_model_path = get_transformer_model_vocab_path(bert_model_name)
            self.writer.write(
                "loaded pretrained model for {} from {}".format(
                    bert_model_name, pretrained_model_path
                )
            )
            bert = AutoModel.from_pretrained(pretrained_model_path)

        self.txt_embeddings = bert.embeddings

        self.img_encoder = VisualEncoder(image_encoder)

        config = Configuration(kwargs)
        self.img_embeddings = ImageBertEmbeddings(
            config,
            img_hidden_sz=self.img_encoder.module.out_dim,
            embeddings=self.txt_embeddings,
        )

        # separate-token between image tokens, default as [SEP]
        self.inter_token_id = kwargs.get(constants.INTER_TOKEN_ID_STR, 102)

        # number of separate-tokens between image tokens, default as 1 for compatibility.
        self.img_token_interval = kwargs.get(constants.IMG_TOKEN_INTERVAL_STR, 1)

        self.encoder = bert.encoder
        self.pooler = bert.pooler

    def build_multimodal_embedding_input(
        self,
        input_txt,
        attention_mask,
        text_token_type,
        cls_id,
        sep_id,
        input_img,
        img_mask=None,
        img_token_type=0,
    ):
        if input_img.ndim == 4:  # bsz, channel, height, width
            bsz, num_images = input_img.size(0), 1
        elif input_img.ndim == 5:  # bsz, num_images, channel, height, width
            bsz, num_images = input_img.size(0), input_img.size(1)
        else:
            raise Exception(f"unknown input image shape:{input_img.shape}")
        length = self.image_encoder.params.num_output_features
        image_seq_length = (
            num_images * length + (num_images - 1) * self.img_token_interval + 2
        )

        img_modality_mask = (
            torch.ones(bsz, image_seq_length + 1)
            .long()
            .to(device=attention_mask.device)
        )
        if img_mask is not None:  # ignore padding visual tokens
            num_visual_tokens = img_mask.sum(axis=1)  # bsz
            attend_visual_tokens = (
                num_visual_tokens * length
                + (num_visual_tokens - 1) * self.img_token_interval
                + 2
            )  # bsz
            # equals to:
            # for idx, attend_visual in enumerate(attend_visual_tokens):
            #     img_modality_mask[idx, attend_visual:] = 0
            idx = attend_visual_tokens.unsqueeze(-1)
            img_modality_mask = 1 - (1 - img_modality_mask).scatter_(
                1, idx, torch.ones_like(idx)
            ).cumsum(1)
        img_modality_mask = img_modality_mask[:, :-1]
        attention_mask = torch.cat(
            [
                img_modality_mask,
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # As of now, Pytorch doesn't support calling self.parameters() within DataParallel, which causes the current
        # issue. Even after fixing that, which was straightforward, Pytorch also doesn't support calling
        # self.ParameterList and self.ParameterDict, which will cause another issue. As Pytorch is moving people
        # away from DataParallel, they are unlikely to fix this anytime soon on their end.
        # In the meantime, we could use DistributedDataParallel instead.
        # see detail:
        # https://github.com/huggingface/transformers/issues/8145#issuecomment-721044942

        # add support for data parallel, and drop support for fp16
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        img_tok = (torch.LongTensor(bsz, image_seq_length).fill_(img_token_type)).to(
            device=input_img.device
        )

        self.img_encoder.check_input(input_img)
        img = self.img_encoder(input_img)
        # B x config.num_output_features x 2048 or B x num_images x
        # config.num_output_features x 2048
        assert len(img.size()) in [
            3,
            4,
        ], f" processed image feature dimension {img.shape}"
        if img.size() == 3:  # [BxNx2048]
            # B x num_images x config.num_output_features x 2048
            img = img.unsqueeze(1)
        img_embed_out = self.img_embeddings(
            img,
            img_tok,
            cls_id,
            sep_id,
            inter_token_id=sep_id,
            img_token_interval=self.img_token_interval,
        )

        if isinstance(text_token_type, int):
            text_token_type = torch.zeros_like(input_txt).fill_(text_token_type)
        txt_embed_out = self.txt_embeddings(input_txt, text_token_type)
        embedding_inputs = {
            "img_embed_out": img_embed_out,
            "txt_embed_out": txt_embed_out,
            "input_attention_mask": extended_attention_mask,
            "image_seq_length": image_seq_length,
            "attention_mask": attention_mask,
        }
        return embedding_inputs

    def forward(
        self,
        input_txt,
        attention_mask,
        segment,
        input_img,
        cls_id,
        sep_id,
        img_mask=None,
    ):
        """
        This combines image with text to have the following input
        observation: [[CLS] <Image-Feature> [SEP] <input_txt>]
        position:    [0, 1, ..., len(Image-Feature) len(Image-Feature) + 1 0, 1, ..., len(input_txt) ]
        token_type:  [0, 0, ..., 0          0,    1, 1, ..., 1]
        attention_mask:
                     [1, 1, ..., 1          1,    1, 1, ..., 0]
        text is padded with zero on attention mask.

        input_txt : size torch.Size([bsz, text_length])
        attention_mask size torch.Size([bsz, text_length])
        segment size torch.Size([bsz, text_length])
        input_img size torch.Size([bsz, channel, height, width]), or torch.Size([bsz, num_images, channel, height,
        width])
        cls_id size torch.Size([bsz])
        sep_id size torch.Size([bsz])
        img_mask: size torch.Size(bsz, num_images), indicate padding.
        """
        embedding_inputs = self.build_multimodal_embedding_input(
            input_txt, attention_mask, segment, cls_id, sep_id, input_img, img_mask
        )
        img_embed_out, txt_embed_out, extended_attention_mask, image_seq_length = (
            embedding_inputs["img_embed_out"],
            embedding_inputs["txt_embed_out"],
            embedding_inputs["input_attention_mask"],
            embedding_inputs["image_seq_length"],
        )

        encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # Bx(IMG+TEXT)xHID
        encoded_layers = self.encoder(
            encoder_input,
            extended_attention_mask,
            head_mask=[None] * len(self.encoder.layer),
        )

        rep_info = self._representations(
            encoded_layers, img_embed_out.shape[1], extended_attention_mask
        )
        pooled_out = self.pooler(encoded_layers[0])
        # add multi-modal embedding
        rep_info["bert_embedding"] = pooled_out
        outputs = (
            pooled_out,  # pooled output
            rep_info,
            encoded_layers[0],  # sentence_out
            image_seq_length,
        )
        return outputs

    def get_adv_parameters(self):
        # return parameters for adversarial genration and training
        # only do adversarial on the word-embeddings, not on others such as
        # positional embedding
        adv_param_names = ["word_embeddings.weight"]
        params = []
        for n, v in self.txt_embeddings.named_parameters():
            if n in adv_param_names:
                params.append(v)
        ret = [
            {"params": p, "name": n, "modality": "text"}
            for n, p in zip(adv_param_names, params)
        ]
        return ret

    def _representations(self, encoded_layers, img_sep_pos, extended_attention_mask):
        r"""
        Extract image & text representations from MMBT.
        Image representations are the average of image segment in the last layer of MMBT.
        Text representations are the average of text segment in the last layer of MMBT.
        Args:
            img_sep_pos: the position where sep_id is located for seperating image modality from text modality
        """
        if self.output_rep_info is False:
            return {}

        bsz = extended_attention_mask.shape[0]
        last_layer = encoded_layers[-1]
        # Note: image_modality_rep & text_modality contains [SEP], which might
        # not accurate.
        image_modality_rep = torch.einsum(
            "bld -> bd", last_layer[:, 1 : img_sep_pos - 1, :]
        ) / (img_sep_pos - 2)
        text_modality = last_layer[:, img_sep_pos:, :]
        text_mask = extended_attention_mask.view(bsz, -1)[:, img_sep_pos:]
        text_modality = torch.einsum("bld, bl -> bd", text_modality, text_mask)
        text_modality_length = torch.einsum("bl -> b", text_mask)
        text_modality = text_modality / text_modality_length.unsqueeze(1)
        # length normalization
        return {
            constants.IMAGE_MODALITY: image_modality_rep,
            constants.TEXT_MODALITY: text_modality,
        }
