# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("mce_loss")
class MultiLabelCategoryCrossEntropyLoss(nn.Module):
    """
    Multi-label cross-entropy loss for multi-label classification:
    performs better than multi-label sigmoid loss, which suffers from
    multi-label class imbalance and threshold tuning problems.

    See detail at:
    https://spaces.ac.cn/archives/7359/comment-page-1

    """

    def __init__(self, **params):
        super().__init__()
        if params is None:
            params = {}
        # support model with multiple CrossEntropyLoss
        self._logit_key = params.pop("logits_key", "logits")

    def forward(self, sample_list, model_output, *args, **kwargs):
        """
        y_true和y_pred的shape一致，y_true的元素非0即1，
        1表示对应的类为目标类，0表示对应的类为非目标类。
        预测阶段输出y_pred大于0的类
        """
        assert "targets" in sample_list
        y_pred = model_output[self._logit_key]
        y_true = sample_list["targets"]
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        loss = neg_loss + pos_loss
        return loss
