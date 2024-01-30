# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch


class SwishFunction(torch.autograd.Function):
    """
    A memory-efficient implementation of Swish function.

    The equation of swish can be writen as:

    .. math::

        y = x * \\sigma(x)
    """

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
