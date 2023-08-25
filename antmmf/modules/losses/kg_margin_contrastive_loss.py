# -- coding: utf-8 --

import torch
import torch.nn.functional as F
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("kg_margin_contrastive")
class KGMarginContrastiveLoss(nn.Module):
    """
    losses config setting:
      - type: kg_margin_contrastive
        params:
            margin_value: 0.9
            contrastive_type: normal
    """

    def __init__(self, margin_value: float = 0.9, contrastive_type: str = "normal"):
        from kgrl.models.pytorch.conf import KGConstranstiveType

        super(KGMarginContrastiveLoss, self).__init__()
        self.contrastive_type = contrastive_type
        self.margin_value = margin_value

        assert self.contrastive_type in [
            KGConstranstiveType.NORMAL,
            KGConstranstiveType.ROTATE_LIKE,
        ], f"contrastive_type can only in {KGConstranstiveType.NORMAL}, {KGConstranstiveType.ROTATE_LIKE}."

    def forward(self, sample_list, model_output):
        from kgrl.models.pytorch.conf import KGConstranstiveType

        assert "targets" in sample_list, "sample_list must have targets"
        assert "pos_score" in model_output, "model_output must have pos_score"
        assert "neg_head_score" in model_output, "model_output must have neg_head_score"
        assert "neg_tail_score" in model_output, "model_output must have neg_tail_score"

        pos_score = model_output["pos_score"]
        neg_head_score = model_output["neg_head_score"]
        neg_tail_score = model_output["neg_tail_score"]

        if self.contrastive_type == KGConstranstiveType.NORMAL:
            loss = torch.mean(
                F.relu(pos_score - neg_tail_score + self.margin_value), dim=-1
            ) + torch.mean(
                F.relu(pos_score - neg_head_score + self.margin_value), dim=-1
            )
        else:

            def loss_compute(item):
                return -torch.mean(torch.log(torch.sigmoid(item) + 1e-12))

            loss = loss_compute(self.margin_value - pos_score)
            loss = loss + loss_compute(neg_head_score - self.margin_value)
            loss = loss + loss_compute(neg_tail_score - self.margin_value)
        return loss
