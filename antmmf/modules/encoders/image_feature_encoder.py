# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import os
import pickle
import torch
import torch.nn.functional as F
from torch import nn
from antmmf.common import configurable
from antmmf.modules.module_registry import ModuleRegistry


class ImageFeatureEncoder(ModuleRegistry):
    """
    A graph encoder register for image feature, all other details can be
    seen from :class:`antmmf.nn.registry.ModuleRegistry`.

    Args:
        encoder_type (str): type of image feature encoder.
    """

    def __init__(self, encoder_type: str, *args, **kwargs):
        # compatible codes, and they will be removed in the future.
        type_mapping = {
            "default": "Identity",
        }
        if encoder_type in type_mapping:
            encoder_type = type_mapping[encoder_type]

        super(ImageFeatureEncoder, self).__init__(encoder_type, *args, **kwargs)


ImageFeatureEncoder.register(nn.Identity)


@ImageFeatureEncoder.register()
class FinetuneFasterRcnnFpnFc7(nn.Module):
    @configurable
    def __init__(
        self,
        in_dim: int,
        weights_file: str,
        bias_file: str,
        model_data_dir: str,
    ):
        super(FinetuneFasterRcnnFpnFc7, self).__init__()

        if not os.path.isabs(weights_file):
            weights_file = os.path.join(model_data_dir, weights_file)
        if not os.path.isabs(bias_file):
            bias_file = os.path.join(model_data_dir, bias_file)
        with open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with open(bias_file, "rb") as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        x = self.lc(image)
        x = F.relu(x)
        return x
