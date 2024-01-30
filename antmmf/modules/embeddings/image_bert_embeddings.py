# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn

from antmmf.common import Configuration, configurable


class ImageBertEmbeddings(nn.Module):
    @configurable
    def __init__(
        self,
        embeddings: Configuration,
        dropout: float,
        hidden_size: int = None,
        hidden_sz: int = None,
        img_hidden_sz: int = None,
    ):
        super(ImageBertEmbeddings, self).__init__()
        # for compatibility
        self.img_embeddings = nn.Linear(
            img_hidden_sz,
            hidden_size or hidden_sz,
        )
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=dropout)

    """
    output embedding has the following information
    content:    CLS <IMAGE1> SEP <IMAGE2> SEP <IMAGE_N> SEP
    position:   0, 1, ..., num_images*LEN(IMAGE) + (num_images-1) + 1
    token_type: t, t, ..., t
    where t is the token type
    These are summed together as the output
    """

    def forward(
        self,
        input_imgs,
        token_type_ids,
        cls_id,
        sep_id,
        inter_token_id=None,
        img_token_interval=1,
    ):
        """
        input_imgs: size of [bsz, num_images, length, image_input_dim]
        token_type_ids: size of [bsz, image_embedding_dim]
        cls_id: [CLS] token id from tokenizer, size of [bsz]
        sep_id: [SEP] token id from tokenizer, size of [bsz]
        """
        if input_imgs.ndim == 3:  # [bsz, length, image_input_dim]
            input_imgs = input_imgs.unsqueeze(1)
        bsz, num_images, length, _ = input_imgs.size()
        # TODO: image feture is pretty weak, given that there is only
        #       image embedding
        # this leaves future performance improvement potential by having more
        # objects detected
        seq_length = (
            num_images * length + (num_images - 1) * img_token_interval + 2
        )  # +2 for CLS and SEP Token

        cls_token_embeds = self.word_embeddings(cls_id).unsqueeze(1)
        sep_token_embeds = self.word_embeddings(sep_id).unsqueeze(1)
        if inter_token_id is None:
            inter_token_embeds = sep_token_embeds
        else:
            inter_id = inter_token_id * cls_id.new_ones(cls_id.size())
            inter_token_embeds = self.word_embeddings(inter_id).unsqueeze(1)

        # bsz, num_images, length, hidden_sz
        imgs_embeddings = self.img_embeddings(input_imgs)

        img_sep_embeds = torch.cat(
            [imgs_embeddings]
            + [inter_token_embeds.unsqueeze(1).repeat(1, imgs_embeddings.size(1), 1, 1)]
            * img_token_interval,
            2,
        ).flatten(1, 2)
        token_embeddings = torch.cat([cls_token_embeds, img_sep_embeds], 1)

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=imgs_embeddings.device
        )
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
