# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch.nn.functional as F


def convert_to_one_hot(expected, output):
    # This won't get called in case of multilabel, only multiclass or binary
    # as multilabel will anyways be multi hot vector
    if output.squeeze().dim() != expected.squeeze().dim() and expected.dim() == 1:
        expected = F.one_hot(expected.long(), num_classes=output.size(-1)).float()
    return expected
