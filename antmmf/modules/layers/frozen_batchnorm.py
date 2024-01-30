# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    It has the same output as the normal nn.BatchNorm2d.
    The batch-size of input images are limited in multi-modal scenerios, freeze the
    statistics of batch-normalization will usually bring boost in performance.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.

    Refer to Detr Implementation:
    https://github.com/facebookresearch/detr/blob/master/models/backbone.py#L19

    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # Take params as buffer instead of nn.parameter. Buffer params will not get updated,
        # but will resume from checkpoint.
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        # peform normal BN:
        # BN(x) = (x - rm)*w/sqrt(rv + eps) + b
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
