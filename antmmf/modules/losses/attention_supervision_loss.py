# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch.nn.functional as F
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("attention_supervision")
class AttentionSupervisionLoss(nn.Module):
    """Loss for attention supervision. Used in case you want to make attentions
    similar to some particular values.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = F.binary_cross_entropy

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Calculates and returns the multi loss.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        context_attentions = model_output["attentions"]
        attention_supervision = sample_list["info"]["attention_supervision"]

        loss = self.loss_fn(
            context_attentions[0],
            attention_supervision.float(),
            weight=attention_supervision.float(),
        )

        # Multiply average loss back with target size to get actual loss
        return loss * attention_supervision.size(1)
