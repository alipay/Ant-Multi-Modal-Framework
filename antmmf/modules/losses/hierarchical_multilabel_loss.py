# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common.registry import registry
from .asymmetric_loss_optimized import AsymmetricLossOptimized


@registry.register_loss("hier_multilabel_loss")
class HierarchicalMultilabelLoss(nn.Module):
    """
    Hierarchical Softmax Loss, used by
    antmmf.modules.decoders.HierarchicalClassifier
    """

    def __init__(self, **params):
        super().__init__()
        if params is None:
            params = {}
        self.gamma_neg = params.pop("gamma_neg", 4)
        self.gamma_pos = params.pop("gamma_pos", 1)
        self.clip = params.pop("clip", 0.05)
        self.disable_torch_grad_focal_loss = params.pop(
            "disable_torch_grad_focal_loss", False
        )
        self.loss_asymetric = AsymmetricLossOptimized(
            gamma_neg=self.gamma_neg,
            gamma_pos=self.gamma_pos,
            clip=self.clip,
            disable_torch_grad_focal_loss=self.disable_torch_grad_focal_loss,
        )

    def forward(self, sample_list, model_output, *args, **kwargs):
        """
        Args:
            hier_logits(list): outputs of antmmf.modules.decoders.HierarchicalClassifier
            hier_label(torch.tensor): [batch_size, len(self.tree.ParamGroup),  n_class], with -1 padded
            hier_param(torch.tensor): [batch_size, len(self.tree.ParamGroup)], with -1 padded
            hier_label_num(torch.tensor): [batch_size, len(self.tree.ParamGroup)], with -1 padded
        Returns:
            torch.tensor: hierarchical loss of the batch
        """
        assert "hier_label" in sample_list and "hier_param" in sample_list
        hier_labels, hier_params = sample_list["hier_label"], sample_list["hier_param"]
        assert "hier_logits" in model_output
        hier_logits = model_output["hier_logits"]

        total_loss = []
        nbz = hier_labels.size(0)
        for batch_idx in range(nbz):
            sample_loss = []
            hier_label, hier_param = hier_labels[batch_idx], hier_params[batch_idx]
            assert "hier_label_num" in sample_list
            hier_labels_num = sample_list["hier_label_num"]
            assert hier_labels_num.shape == hier_params.shape
            hier_label_num_bz = hier_labels_num[batch_idx]
            for l, p, n in zip(hier_label, hier_param, hier_label_num_bz):
                if p == -1:  # skip padding labels
                    continue
                logit = hier_logits[p][batch_idx]
                if n <= 0:  # skip padding labels
                    label_bit = torch.zeros(
                        (1, logit.shape[0]), device=logit.device, dtype=torch.float
                    )
                else:
                    label_index = l[0:n]
                    label_bit = torch.zeros_like(logit).scatter_(
                        0, label_index.long(), 1
                    )
                loss = self.loss_asymetric.asy_loss(
                    logit.unsqueeze(0), label_bit.unsqueeze(0)
                )
                sample_loss.append(loss)
            total_loss.append(torch.stack(sample_loss, 0).sum())

        total_loss = torch.stack(total_loss, 0)
        return total_loss.mean(dim=0)
