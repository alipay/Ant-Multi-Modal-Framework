# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from antmmf.common.registry import registry
from antmmf.modules.metrics.mce_accuracy import MCEAccuracy


@registry.register_metric("rule_multi_label_metric")
class RMCEAccuracy(MCEAccuracy):
    """Metric for calculating multi-label accuracy
    with supporting of copy rules and soft label.
    https://spaces.ac.cn/archives/7359/comment-page-1
    """

    def __init__(self, name: str = "rule_multi_label_metric"):
        super(RMCEAccuracy, self).__init__(name=name)

    def _get_pred_label(self, prob):
        return torch.where(prob >= 0.5, 1, 0).long()

    def _mce_calculate_acc_and_rec(self, sample_list, model_output, *args, **kwargs):
        """Calculate mce accuracy & recall."""
        prob = model_output["prob"]
        prob = prob.view(-1, prob.size(-1))
        targets = sample_list["targets"].long()
        assert prob.shape == targets.shape
        # acc = TP / TP + FP
        # output label where logits >= 0
        predict = self._get_pred_label(prob)
        # only count non-zero labels
        TP = (torch.where(targets > 0.5, 1, -1) == predict).sum()
        TP_AND_FP = predict.sum()
        TP_AND_FN = targets.sum()
        return TP, TP_AND_FP, TP_AND_FN
