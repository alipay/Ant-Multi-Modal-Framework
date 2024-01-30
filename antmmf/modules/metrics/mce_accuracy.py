# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import numpy as np
import torch
from antmmf.common.registry import registry
from antmmf.modules.metrics.base_metric import BaseMetric
from sklearn.metrics import precision_recall_curve


@registry.register_metric("multi_label_metric")
class MCEAccuracy(BaseMetric):
    """Metric for calculating multi-label accuracy.
    https://spaces.ac.cn/archives/7359/comment-page-1
    """

    def __init__(self, name: str = "multi_label_metric"):
        super().__init__(name)
        self.reset()

    def reset(self):
        self._predict_correct = 0
        self._predict_total = 0
        self._gt_total = 0

    def _get_pred_label(self, logits):
        return torch.where(logits >= 0, 1, 0).long()

    def _mce_calculate_acc_and_rec(self, sample_list, model_output, *args, **kwargs):
        """Calculate mce accuracy & recall."""
        logits = model_output["logits"]
        logits = logits.view(-1, logits.size(-1))
        targets = sample_list["targets"].long()
        assert logits.shape == targets.shape
        # acc = TP / TP + FP
        # output label where logits >= 0
        predict = self._get_pred_label(logits)
        # only count non-zero labels
        TP = (torch.where(targets > 0, 1, -1) == predict).sum()
        TP_AND_FP = predict.sum()
        TP_AND_FN = targets.sum()
        return TP, TP_AND_FP, TP_AND_FN

    def calculate(self, sample_list, model_output, *args, **kwargs):
        TP, TP_AND_FP, TP_AND_FN = self._mce_calculate_acc_and_rec(
            sample_list, model_output, *args, **kwargs
        )
        pre = TP / (TP_AND_FP + 1e-20)
        rec = TP / (TP_AND_FN + 1e-20)
        f1 = 2 * pre * rec / (pre + rec)
        return {
            "multi_precision": pre.clone().detach(),
            "multi_recall": rec.clone().detach(),
            "multi_f1": f1.clone().detach(),
        }

    def collect(self, sample_list, model_output, *args, **kwargs):
        TP, TP_AND_FP, TP_AND_FN = self._mce_calculate_acc_and_rec(
            sample_list, model_output, *args, **kwargs
        )
        predict_correct = TP
        predict_total = TP_AND_FP
        gt_total = TP_AND_FN
        self._predict_correct += predict_correct.detach().cpu()
        self._predict_total += predict_total.detach().cpu()
        self._gt_total += gt_total.detach().cpu()

    def summarize(self, *args, **kwargs):
        mce_precision = np.around(
            (self._predict_correct / (self._predict_total + 1e-20)).numpy(), 4
        )
        mce_recall = np.around(
            (self._predict_correct / (self._gt_total + 1e-20)).numpy(), 4
        )
        mce_f1 = 2 * mce_precision * mce_recall / (mce_precision + mce_recall)
        return {
            "multi_precision": torch.tensor(mce_precision, dtype=torch.float),
            "multi_recall": torch.tensor(mce_recall, dtype=torch.float),
            "multi_f1": torch.tensor(mce_f1, dtype=torch.float),
        }


@registry.register_metric("multi_label_threshold_metric")
class MCEThreshMetric(MCEAccuracy):
    """Metric for calculating multi-label accuracy.
    Dynamically calculate the threshold, and calculate the accuracy based on the optimal threshold.
    The optimal threshold is written to the file for prediction and infer.
    https://spaces.ac.cn/archives/7359/comment-page-1
    """

    def __init__(self, *args, **kwargs):
        super().__init__("multi_label_threshold_metric")
        self.threshold_path = kwargs.get("threshold_path", "./thresholds.txt")
        self.label_decoder = kwargs.get("label_decoder", [])
        self.reset()

    def reset(self):
        self.probs = torch.tensor([], dtype=torch.float64, device="cpu")
        self.targets = torch.tensor([], dtype=torch.float64, device="cpu")

    def _get_pred_label(self, logits):
        return torch.where(torch.sigmoid(logits) >= 0, 1, 0).long()

    def collect(self, sample_list, model_output, *args, **kwargs):
        # logits = model_output["logits"]
        logits = model_output["logits"]
        targets = sample_list["targets"].long()
        assert logits.shape == targets.shape
        self.probs = torch.cat(
            (self.probs, torch.sigmoid(logits).detach().cpu().double())
        )
        self.targets = torch.cat((self.targets, targets.detach().cpu().double()))

    def summarize(self, *args, **kwargs):
        """
        Dynamically calculate the threshold, and calculate the accuracy based on the optimal threshold.
        The optimal threshold is written to the file for prediction and infer.
        """
        probs = self.probs.cpu().transpose(1, 0)
        targets = self.targets.cpu().transpose(1, 0)
        f_thresholds = open(self.threshold_path, "w")
        cls_thresholds = {}
        cls_prf = {}
        for i in range(len(self.label_decoder) - 1):
            # Calculate the optimal threshold by category
            cls_name = self.label_decoder[i]
            precision, recall, thresholds = precision_recall_curve(targets[i], probs[i])
            f1_scores = 2 * recall * precision / (recall + precision)
            best_threshold = thresholds[np.argmax(f1_scores)]
            if np.isnan(best_threshold):
                best_threshold = 0.5
            cls_thresholds[i] = float(best_threshold)
            f_thresholds.write(cls_name + "\t" + str(best_threshold) + "\n")
            best_precision = precision[np.argmax(f1_scores)]
            best_recall = recall[np.argmax(f1_scores)]
            best_f1 = np.max(f1_scores)
            if np.isnan(best_precision):
                best_precision = 0.0
            if np.isnan(best_recall):
                best_recall = 0.0
            if np.isnan(best_f1):
                best_f1 = 0.0
            cls_prf[self.label_decoder[i]] = [
                sum(targets[i]),
                float(best_precision),
                float(best_recall),
                float(best_f1),
            ]
        num_total, num_pred, num_hit = 0.0, 0.0, 0.0
        for i in range(len(cls_prf)):
            _total = cls_prf[self.label_decoder[i]][0]
            _hit = float(int(_total * cls_prf[self.label_decoder[i]][2]))
            _pred = (
                _hit / cls_prf[self.label_decoder[i]][1]
                if cls_prf[self.label_decoder[i]][1]
                else 0.0
            )
            num_total += _total
            num_hit += _hit
            num_pred += _pred
        p = num_hit / num_pred if num_pred else 0.0
        r = num_hit / num_total if num_total else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0.0 else 0.0
        cls_prf["total"] = [num_total, p, r, f1]
        # print('cls_prf:', cls_prf)
        # cls_prf["multi_precision"] = p
        # cls_prf["multi_recall"] = r
        # cls_prf["multi_f1"] = f1
        return {"multi_precision": p, "multi_recall": r, "multi_f1": f1}
