# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch


class IdentitySegmentConsensus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class AvgSegmentConsensus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.input_shape = input_tensor.size()
        output = input_tensor.mean(dim=1, keepdim=True)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.input_shape
        grad_in = grad_output.expand(shape) / float(shape[1])
        return grad_in


class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != "rnn" else "identity"
        self.dim = dim

    def forward(self, input):
        """
        see issue at: https://discuss.pytorch.org/t/legacy-autograd-function-with-non-static-forward-method-is-deprecated-and-will-be-removed-in-1-3/69869 # noqa
        Legacy autograd function with non-static forward method is deprecated and removed in 1.3
        """
        segmentConsensus = None
        if self.consensus_type == "avg":
            segmentConsensus = AvgSegmentConsensus
        elif self.consensus_type == "identity":
            segmentConsensus = IdentitySegmentConsensus
        assert issubclass(segmentConsensus, torch.autograd.Function)
        return segmentConsensus.apply(input)
