# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from typing import Optional  # noqa
from antmmf.modules.classifier import ClassifierLayer


@ClassifierLayer.register()
class LogitClassifier(nn.Module):
    """
    The simplest classifier for multi-modality (only image and text) classification, and
    :external:py:func:`weight_norm <torch.nn.utils.weight_norm>` is used in hidden layers.

    Args:
        in_dim (int): dimension of input feature of image and text.
        out_dim (int): dimension of output logits, and it is equal to the number of classes.
        text_hidden_dim (int): dimension of hidden layer of text.
        img_hidden_dim (int): dimension of hidden layer of image.
        pretrained_image (Optional[np.ndarray]): pretrained weights of image classification layer.
        pretrained_text (Optional[np.ndarray]): pretrained weights of text classification layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        text_hidden_dim: int,
        img_hidden_dim: int,
        pretrained_image: np.ndarray = None,
        pretrained_text: np.ndarray = None,
    ):
        super(LogitClassifier, self).__init__()

        self.f_o_text = nn.utils.weight_norm(nn.Linear(in_dim, text_hidden_dim), dim=-1)
        self.f_o_image = nn.utils.weight_norm(nn.Linear(in_dim, img_hidden_dim), dim=-1)
        self.linear_text = nn.Linear(text_hidden_dim, out_dim)
        self.linear_image = nn.Linear(img_hidden_dim, out_dim)

        if pretrained_image is not None and pretrained_text is not None:
            self.linear_text.weight.data.copy_(torch.from_numpy(pretrained_text))

        if pretrained_image is not None and pretrained_image is not None:
            self.linear_image.weight.data.copy_(torch.from_numpy(pretrained_image))

    def forward(self, joint_embedding):
        text_val = self.linear_text(F.relu_(self.f_o_text(joint_embedding)))
        image_val = self.linear_image(F.relu(self.f_o_image(joint_embedding)))
        logit_value = text_val + image_val

        return logit_value
