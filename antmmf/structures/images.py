# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

"""
ImageList, a storage of images and their attributes.
Many codes are copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/image_list.py
"""

import torch
from typing import List

from antmmf.structures.utils import create_image_tensor_from_images
from antmmf.structures import SizedDataStructure


class ImageList(SizedDataStructure):
    """
    This structure stores a list of images.
    It can be construct by `from_tensors`, which will padding the image to the maximum sizes
    and record the sizes of each image. Furthermore, `from_tensors` can be make the sizes of
    the images be divisible by size_divisibility.
    """

    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
    ) -> "ImageList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
                to the same shape with `pad_value`.
            size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
                the common height and width is divisible by `size_divisibility`.
                This depends on the model and many models need a divisibility of 32.
            pad_value (float): value to pad

        Returns:
            ImageList:
        """
        batched_images, _, image_sizes = create_image_tensor_from_images(
            images=tensors,
            size_divisibility=size_divisibility,
            pad_value=pad_value,
        )
        image_list = ImageList(batched_images.contiguous())
        image_list.image_sizes = image_sizes
        return image_list
