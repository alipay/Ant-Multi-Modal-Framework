# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import torch.nn.functional as F
from torch import nn
from antmmf.modules.module_registry import ModuleRegistry
from .linear import Linear


class TransformLayer(ModuleRegistry):
    def __init__(self, transform_type, in_dim, out_dim, **kwargs):
        type_mapping = {
            "linear": "Linear",
            "conv": "ConvTransform",
        }
        if transform_type in type_mapping:
            transform_type = type_mapping[transform_type]

        super(TransformLayer, self).__init__(transform_type, in_dim, out_dim, **kwargs)

        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


TransformLayer.register(Linear)


@TransformLayer.register()
class ConvTransform(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(ConvTransform, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=hidden_dim, kernel_size=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=out_dim, kernel_size=1
        )
        self.out_dim = out_dim

    def forward(self, x):
        if len(x.size()) == 3:  # N x k xdim
            # N x dim x k x 1
            x_reshape = torch.unsqueeze(x.permute(0, 2, 1), 3)
        elif len(x.size()) == 2:  # N x dim
            # N x dim x 1 x 1
            x_reshape = torch.unsqueeze(torch.unsqueeze(x, 2), 3)
        else:
            raise NotImplementedError(
                f"invalid input with shape of {x.shape}, only 2d and 3d tensor are available."
            )

        iatt_conv1 = self.conv1(x_reshape)  # N x hidden_dim x * x 1
        iatt_relu = F.relu_(iatt_conv1)
        iatt_conv2 = self.conv2(iatt_relu)  # N x out_dim x * x 1

        if len(x.size()) == 3:
            iatt_conv3 = torch.squeeze(iatt_conv2, 3).permute(0, 2, 1)
        elif len(x.size()) == 2:
            iatt_conv3 = torch.squeeze(torch.squeeze(iatt_conv2, 3), 2)
        else:
            raise NotImplementedError

        return iatt_conv3
