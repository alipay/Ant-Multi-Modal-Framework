# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from typing import Tuple, List
from antmmf.structures.utils import create_image_tensor_from_images
from antmmf.structures.base import SizedDataStructure


class NestedTensor(SizedDataStructure):
    """
    NestedTensor provides an easy way to batch images of different sizes with
    padding mask, which is often the case for models using grid features.

    For resnet used for classification, images are rescaled to 224 x 224, which
    will result in very low performance due to low resolution, see detail at:
    In Defense of Grid Features for Visual Question Answering:
    https://arxiv.org/abs/2001.03615


    It is inspired from Detr:
    https://github.com/facebookresearch/detr/blob/master/util/misc.py#L283-L328
    """

    def __init__(self, tensors, mask=None):
        super(NestedTensor, self).__init__(tensors)
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs) -> "NestedTensor":
        """
        Move `tensors` and `mask` to cuda or cpu, or convert data type, its function is same
        as function of :func:`to` of :external:py:class:`torch.Tensor`.

        Returns:
            NestedTensor: A deep copied NestedTensor whose `tensors` and `mask` are processed.
        """
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def decompose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose NestedTensor to a tuple of tensors

        Returns:
            tensors, mask (torch.Tensor):
        """
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list: List[torch.Tensor]) -> "NestedTensor":
        """
        Create NestedTensor from image tensors.

        Args:
            tensor_list (torch.Tensor): a list of images :external:py:class:`Tensor <torch.Tensor>`.

        Returns:
            NestedTensor:
        """
        tensor, mask, _ = create_image_tensor_from_images(tensor_list, with_mask=True)
        return cls(tensor, mask)

    def __repr__(self):
        return repr(self.tensors)
