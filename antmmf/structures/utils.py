# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import torch.nn.functional as F
from typing import List, Union, Optional, Tuple


def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Tracing friendly way to cast tensor to another tensor's device. Device will be treated
    as constant during tracing, scripting the casting process as whole can workaround this issue.
    """
    return src.to(dst.device)


def shapes_to_tensor(
    x: List[int], device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Turn a list of integer scalars or integer Tensor scalars into a vector,
    in a way that's both traceable and scriptable.
    In tracing, `x` should be a list of scalar Tensor, so the output can trace to the inputs.
    In scripting or eager, `x` should be a list of int.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        assert all(
            [isinstance(t, torch.Tensor) for t in x]
        ), "Shape should be tensor during tracing!"
        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)


def create_image_tensor_from_images(
    images: List[torch.Tensor],
    size_divisibility: int = 0,
    pad_value: Union[int, float, bool] = 0.0,
    with_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple]]:
    """
    Args:
        images: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
            (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
            to the same shape with `pad_value`.
        size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
            the common height and width is divisible by `size_divisibility`.
            This depends on the model and many models need a divisibility of 32.
        pad_value (float): value to pad
        with_mask (bool): whether return the padding-mask or not, default is False

    Returns:
        (torch.Tensor, torch.Tensor, List): `batched_images`, `batched_masks`, `image_sizes`
    """
    assert len(images) > 0
    assert isinstance(images, (tuple, list))
    for t in images:
        assert isinstance(t, torch.Tensor), type(t)
        assert t.shape[:-2] == images[0].shape[:-2], t.shape

    image_sizes = [(im.shape[-2], im.shape[-1]) for im in images]
    image_sizes_tensor = [shapes_to_tensor(x) for x in image_sizes]
    max_size = torch.stack(image_sizes_tensor).max(0).values
    if size_divisibility > 1:
        stride = size_divisibility
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size + stride - 1) // stride * stride

    # handle weirdness of scripting and tracing ...
    if torch.jit.is_scripting():
        max_size: List[int] = max_size.to(dtype=torch.long).tolist()
    else:
        if torch.jit.is_tracing():
            image_sizes = image_sizes_tensor

    batched_masks = None

    if len(images) == 1:
        # This seems slightly (2%) faster.
        # TODO: check whether it's faster for multiple images as well
        image_size = image_sizes[0]
        padding_size = [
            0,
            max_size[-1] - image_size[1],
            0,
            max_size[-2] - image_size[0],
        ]
        batched_images = F.pad(images[0], padding_size, value=pad_value).unsqueeze_(0)
        if with_mask:
            batched_masks = images[0].new_zeros(
                (1, max_size[-2], max_size[-1]), dtype=torch.bool
            )
            batched_masks[0, image_size[0] :, image_size[1] :] = True
    else:
        # max_size can be a tensor in tracing mode, therefore convert to list
        batch_shape = [len(images)] + list(images[0].shape[:-2]) + list(max_size)
        device = (
            None
            if torch.jit.is_scripting()
            else ("cpu" if torch.jit.is_tracing() else None)
        )
        batched_images = images[0].new_full(batch_shape, pad_value, device=device)
        batched_images = move_device_like(batched_images, images[0])
        for img, pad_img in zip(images, batched_images):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
        if with_mask:
            batched_masks = torch.zeros(
                (len(images), max_size[-2], max_size[-1]),
                dtype=torch.bool,
                device=device,
            )
            for img, pad_img in zip(images, batched_masks):
                pad_img[..., img.shape[-2] :, img.shape[-1] :] = True

    return batched_images, batched_masks, image_sizes
