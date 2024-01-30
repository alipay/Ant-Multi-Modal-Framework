# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common import configurable
from antmmf.modules.attention import AttentionLayer


class ImageEmbedding(nn.Module):
    """
    parameters:

    input:
    image_feat_variable: [batch_size, num_location, image_feat_dim]
    or a list of [num_location, image_feat_dim]
    when using adaptive number of objects
    question_embedding:[batch_size, txt_embeding_dim]

    output:
    image_embedding:[batch_size, image_feat_dim]


    """

    @configurable
    def __init__(self, img_dim, question_dim, **kwargs):
        super(ImageEmbedding, self).__init__()

        self.image_attention_model = AttentionLayer(img_dim, question_dim, **kwargs)
        self.out_dim = self.image_attention_model.out_dim

    def forward(self, image_feat_variable, question_embedding, image_dims, extra={}):
        # N x K x n_att
        attention = self.image_attention_model(
            image_feat_variable, question_embedding, image_dims
        )
        att_reshape = attention.permute(0, 2, 1)

        order_vectors = getattr(extra, "order_vectors", None)

        if order_vectors is not None:
            image_feat_variable = torch.cat(
                [image_feat_variable, order_vectors], dim=-1
            )
        tmp_embedding = torch.bmm(
            att_reshape, image_feat_variable
        )  # N x n_att x image_dim
        batch_size = att_reshape.size(0)
        image_embedding = tmp_embedding.view(batch_size, -1)

        return image_embedding, attention
