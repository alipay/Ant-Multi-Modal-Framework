# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common.registry import registry
from typing import List, Dict
from .losses import AntMMFLoss


@registry.register_loss("multi")
class MultiLoss(nn.Module):
    """A loss for combining multiple losses with weights.

    Args:
        losses (List(Dict)): A list containing parameters for each different loss
                             and their weights.

    Example::

        # MultiLoss works with config like below where each loss's params and
        # weights are defined
        losses:
        - type: multi
          params:
          - type: logit_bce
            weight: 0.3
            params: {}
          - type: attention_supervision
            weight: 0.7
            params: {}

    """

    def __init__(self, losses: List[Dict], use_uncertainty_weight: bool = False):
        super().__init__()
        self.losses = []
        self.losses_weights = []
        self.task_names = []
        self.writer = registry.get("writer")

        self.loss_names = []
        # uncertainty weight is used to balance losses from multi task, refer to: https://arxiv.org/abs/1705.07115
        self.use_uncertainty_weight = use_uncertainty_weight

        for loss_params in losses:
            self.loss_names.append(loss_params["type"])
            loss_fn = AntMMFLoss(loss_params)
            loss_weight = loss_params.get("weight", {})
            task_name = loss_params.get("task_name", None)
            if task_name is not None:
                self.task_names.append(task_name)
            self.losses.append(loss_fn)
            self.losses_weights.append(loss_weight)

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Calculates and returns the multi loss.

        Args:
            sample_list (SampleList): SampleList containing `attentions` attribute.
            model_output (Dict): Model output containing `attention_supervision`
                                 attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        # multitask model will contain uncertainty_weight if use_uncertainty_weight is true.
        # refer to antmmf/models/multitask_model.py
        uncertainty_weight = model_output.get("uncertainty_weight", None)

        multitask = len(self.task_names) == len(self.losses)

        loss = 0
        for idx, loss_fn in enumerate(self.losses):
            weighted_loss = 0
            if multitask:
                sample_list["targets"] = getattr(
                    sample_list, "{}_targets".format(self.task_names[idx])
                )
                model_output["logits"] = model_output[
                    "{}_logits".format(self.task_names[idx])
                ]
            value = loss_fn(sample_list, model_output, *args, **kwargs)
            if value is not None:
                if isinstance(value, dict):
                    for k, v in value.items():
                        weighted_loss = (
                            torch.exp(-uncertainty_weight[idx]) * v
                            + uncertainty_weight[idx]
                            if self.use_uncertainty_weight
                            else self.losses_weights[idx] * v
                        )
                else:
                    weighted_loss = (
                        torch.exp(-uncertainty_weight[idx]) * value
                        + uncertainty_weight[idx]
                        if self.use_uncertainty_weight
                        else self.losses_weights[idx] * value
                    )
            loss = loss + weighted_loss

        if not isinstance(loss, torch.Tensor):
            loss = torch.zeros(1)
        return loss
