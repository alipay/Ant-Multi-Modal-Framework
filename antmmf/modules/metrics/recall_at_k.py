# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import numpy as np
import collections.abc

from antmmf.common.registry import registry
from antmmf.modules.metrics import BaseMetric


def score_to_ranks(scores):
    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # convert from ranked_idx to ranks
    ranks = torch.zeros_like(ranked_idx)
    ranks.scatter_(
        1,
        ranked_idx,
        torch.arange(ranked_idx.shape[1], device=ranked_idx.device).repeat(
            ranks.shape[0], 1
        ),
    )
    ranks += 1
    return ranks


def get_gt_ranks(ranks, ans, ans_is_index=False, allow_padding=False, padding_value=-1):
    # ans is index, like [0, 2]
    if ans_is_index:
        # support multiple answers with padding for each gt
        if allow_padding:
            rank_ind, c_idx = torch.where(ans != padding_value)
            ans_ind = ans[rank_ind, c_idx]
        else:
            ans_ind = ans
            # indicate rank index of each ans
            rank_ind = torch.arange(ans_ind.size(0))
    # ans is one-hot tensor, like [[1, 0, 0], [0, 0, 1]]
    else:
        _, ans_ind = ans.max(dim=1)
        ans_ind = ans_ind.view(-1)
        # indicate rank index of each ans
        rank_ind = torch.arange(ans_ind.size(0))

    gt_ranks = torch.LongTensor(ans_ind.size(0))  # groundtruth rank

    for i, rank_idx in enumerate(rank_ind):
        gt_ranks[i] = int(ranks[rank_idx, ans_ind[i].long()])
    return gt_ranks


@registry.register_metric("recall@k")
class RecallAtK(BaseMetric):
    def __init__(self, name="recall@k", *args, **kwargs):
        super().__init__(name)
        self._k = kwargs.get("k")
        if self._k and not isinstance(self._k, collections.abc.Sequence):
            self._k = [self._k]
        self.ans_is_index = kwargs.get("ans_is_index", False)
        self.allow_padding = kwargs.get("allow_padding", False)
        self.reset()

    def get_ranks(self, sample_list, model_output, *args, **kwargs):
        output = model_output["logits"]
        expected = sample_list["targets"]

        ranks = score_to_ranks(output)
        gt_ranks = get_gt_ranks(
            ranks,
            expected,
            ans_is_index=self.ans_is_index,
            allow_padding=self.allow_padding,
            **kwargs,
        )

        return gt_ranks.float()

    def _calculate(self, ranks):
        assert self._k is not None
        if not isinstance(self._k, collections.abc.Sequence):
            self._k = [self._k]
        metric_dict = {}
        for k in self._k:
            recall = float(np.sum(np.less_equal(ranks, k))) / len(ranks)
            metric_dict[self.name + f"_{k}"] = recall
        return metric_dict

    def calculate(self, sample_list, model_output, *args, **kwargs):
        ranks = self.get_ranks(sample_list, model_output)
        return self._calculate(ranks.cpu().detach().numpy())

    def reset(self):
        self._ranks = np.array([])

    def collect(self, sample_list, model_output, *args, **kwargs):
        ranks = self.get_ranks(sample_list, model_output)
        self._ranks = np.concatenate((self._ranks, ranks.cpu().detach().numpy()))

    def summarize(self, *args, **kwargs):
        return self._calculate(self._ranks)
