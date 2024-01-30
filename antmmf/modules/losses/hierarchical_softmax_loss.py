# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("hier_loss")
class HierarchicalSoftmaxLoss(nn.Module):
    """
    Hierarchical Softmax Loss, used by
    antmmf.modules.decoders.HierarchicalClassifier
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, sample_list, model_output, *args, **kwargs):
        """
        Args:
            hier_logits(list): outputs of antmmf.modules.decoders.HierarchicalClassifier
            hier_label(torch.tensor): [batch_size, len(self.tree.ParamGroup)], with -1 padded
            hier_param(torch.tensor): [batch_size, len(self.tree.ParamGroup)], with -1 padded

        Returns:
            torch.tensor: hierarchical loss of the batch
        """
        assert "hier_label" in sample_list and "hier_param" in sample_list
        hier_labels, hier_params = sample_list["hier_label"], sample_list["hier_param"]
        assert (
            hier_labels.shape == hier_params.shape
        ), "shape of hier_label and hier_param must match"

        # Note: not all parameters have gradients during hierarchical softmax loss calculation.
        # For DistributedDataParallel training mode, the model's output will be checked, which will
        # cause RuntimeError: Expected to have finished reduction in the prior iteration before
        # starting a new one. Removing 'hier_logits' from model_output is a walk around.
        assert "hier_logits" in model_output
        hier_logits = model_output.pop("hier_logits")

        total_loss = []
        nbz = hier_labels.size(0)
        for batch_idx in range(nbz):
            sample_loss = []
            hier_label, hier_param = hier_labels[batch_idx], hier_params[batch_idx]
            for l, p in zip(hier_label, hier_param):
                if l == -1:  # skip padding labels
                    continue
                logit = hier_logits[p][batch_idx]
                loss = self.loss(logit.unsqueeze(0), l.unsqueeze(0))
                sample_loss.append(loss)
            total_loss.append(torch.stack(sample_loss, 0).sum())

        total_loss = torch.stack(total_loss, 0)
        return total_loss.mean(dim=0)
