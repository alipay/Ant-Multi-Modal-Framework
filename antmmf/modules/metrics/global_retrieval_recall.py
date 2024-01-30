# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from collections import defaultdict

import numpy as np
import torch

from antmmf.common.registry import registry
from antmmf.modules.metrics.base_metric import BaseMetric


def _compute_retrieval_metrics(x):
    """
    compute global retrieval recall metrics
    after collect all similarity scores
    """
    if isinstance(x, np.float32):
        x = np.array([[x]])
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    return ind


def _cal_sym_recall(sim_matrix, t2v, v2t):
    # t2v retrieval metrics
    text_count, visual_count = sim_matrix.shape[0], sim_matrix.shape[1]
    r1_stat, r5_stat, r10_stat = 0, 0, 0
    top10_t2v_ind = np.argsort(-sim_matrix, axis=1)
    t2v_gt_rank = np.zeros(text_count, dtype=np.int32)
    for idx, pred_t2v_ind in enumerate(top10_t2v_ind):
        ground_truth_ids = list(set(t2v[idx]))
        t2v_gt_rank[idx] = min(
            [np.where(pred_t2v_ind == gt_id)[0].min() for gt_id in ground_truth_ids]
        )
        if any([idx in pred_t2v_ind[:1] for idx in ground_truth_ids]):
            r1_stat += 1
        if any([idx in pred_t2v_ind[:5] for idx in ground_truth_ids]):
            r5_stat += 1
        if any([idx in pred_t2v_ind[:10] for idx in ground_truth_ids]):
            r10_stat += 1
    # the higher score, the better
    t2v_r1, t2v_r5, t2v_r10 = (
        r1_stat * 1.0 / text_count,
        r5_stat * 1.0 / text_count,
        r10_stat * 1.0 / text_count,
    )

    # t2v retrieval metrics
    r1_stat, r5_stat, r10_stat = 0, 0, 0
    top10_v2t_ind = np.argsort(-sim_matrix.T, axis=1)
    v2t_gt_rank = np.zeros(visual_count, dtype=np.int32)
    for idx, pred_v2t_ind in enumerate(top10_v2t_ind):
        ground_truth_ids = list(set(v2t[idx]))
        v2t_gt_rank[idx] = min(
            [np.where(pred_v2t_ind == gt_id)[0].min() for gt_id in ground_truth_ids]
        )
        if any([idx in pred_v2t_ind[:1] for idx in ground_truth_ids]):
            r1_stat += 1
        if any([idx in pred_v2t_ind[:5] for idx in ground_truth_ids]):
            r5_stat += 1
        if any([idx in pred_v2t_ind[:10] for idx in ground_truth_ids]):
            r10_stat += 1
    # the higher score, the better
    v2t_r1, v2t_r5, v2t_r10 = (
        r1_stat * 1.0 / visual_count,
        r5_stat * 1.0 / visual_count,
        r10_stat * 1.0 / visual_count,
    )

    return {
        "t2v-mean_recall": (t2v_r1 + t2v_r5 + t2v_r10) / 3.0,
        "t2v-r@1": t2v_r1,
        "t2v-r@5": t2v_r5,
        "t2v-r@10": t2v_r10,
        "t2v-mr": np.median(t2v_gt_rank) + 1,
        "v2t-mean_recall": (v2t_r1 + v2t_r5 + v2t_r10) / 3.0,
        "v2t-r@1": v2t_r1,
        "v2t-r@5": v2t_r5,
        "v2t-r@10": v2t_r10,
        "v2t-mr": np.median(v2t_gt_rank) + 1,
    }


def _cal_recall(ind, epsilon=1e-10):
    ind = _compute_retrieval_metrics(ind)

    def _recall(topk):
        return float(np.sum(ind < int(topk)) / (len(ind) + epsilon))

    return {
        "mr": np.median(ind) + 1,
        "r@1": _recall(1),
        "r@5": _recall(5),
        "r@10": _recall(10),
    }


@registry.register_metric("global_retrieval_recall")
class GlobalRetrievalRecall(BaseMetric):
    """
    Calculate similarity-based bi-modal retrieval performance.
    Process the similarity matrix of two madalities(text & visual).
    Evaluate on the whole val/test set.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name=kwargs.get("name", "global_retrieval_recall"))
        self._simi_logit_key = kwargs.get("simi_logit_key")
        self._ind = dict([(k, None) for k in self._simi_logit_key])
        self.gt_t2v = dict()  # List[List[int]]
        self.gt_v2t = dict()  # List[List[int]]

    def reset(self):
        for simi_level in self._simi_logit_key:
            self._ind[simi_level] = None

    def collect(
        self, sample_list, model_output, idx_t, idx_v, t2v=None, v2t=None, **kwargs
    ):
        """
        Args:
            model_output (Dict): Dict returned by model, that contains similarity scores
        """
        row_no = idx_t

        if t2v is not None and idx_t not in self.gt_t2v:
            self.gt_t2v[idx_t] = t2v
        if v2t is not None and idx_v not in self.gt_v2t:
            self.gt_v2t[idx_v] = v2t

        # init the similarity matrix
        for simi_level in self._simi_logit_key:
            if self._ind[simi_level] is None:
                self._ind[simi_level] = defaultdict(list)

        # fill the similarity matrix
        for simi_level in self._simi_logit_key:
            if simi_level not in model_output:
                continue
            simi_score = model_output[simi_level].cpu().numpy()
            self._ind[simi_level][row_no].append(simi_score)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        score_dict = dict()
        for logit_key in self._simi_logit_key:
            if logit_key not in model_output:
                continue
            simi_score = model_output[logit_key]
            simi_matrix = simi_score  # train, cross-simi matrix

            if simi_matrix.size(0) == simi_matrix.size(1):
                # metric can only be calculated for square matrix
                metric_dict = _cal_recall(simi_matrix)
            else:
                # corner case when construct global simi-matrix, in which
                # case batch-wise metrics are ignored
                metric_dict = {
                    "mr": 0.0,
                    "r@1": 0.0,
                    "r@5": 0.0,
                    "r@10": 0.0,
                }
            for name, val in metric_dict.items():
                lv = "{}_{}".format(logit_key, name)
                score_dict[lv] = torch.tensor(val, dtype=torch.float64)
        return score_dict

    def summarize(self, *args, **kwargs):
        score_dict = dict()
        # get gt mapping
        t2v = [a for x in sorted(self.gt_t2v.items(), key=lambda x: x[0]) for a in x[1]]
        v2t = [a for x in sorted(self.gt_v2t.items(), key=lambda x: x[0]) for a in x[1]]

        for logit_key, logit_dict in self._ind.items():
            if len(logit_dict) == 0:
                continue
            # construct complete matrix
            simi_matrix = np.concatenate(
                [
                    np.concatenate(v, 1)
                    for k, v in sorted(logit_dict.items(), key=lambda x: x[0])
                ],
                0,
            )
            for name, val in _cal_sym_recall(simi_matrix, t2v, v2t).items():
                lv = "{}_{}".format(logit_key, name)
                score_dict[lv] = torch.tensor(val, dtype=torch.float64)
        return score_dict
