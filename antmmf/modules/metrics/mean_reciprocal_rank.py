# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import numpy as np
from antmmf.common.registry import registry
from antmmf.modules.metrics.recall_at_k import RecallAtK


@registry.register_metric("mean_rr")
class MeanReciprocalRank(RecallAtK):
    """
    Calculate reciprocal of mean rank..

    **Key**: ``mean_rr``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(kwargs.get("name", "mean_rr"), *args, **kwargs)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Mean Reciprocal Rank and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Mean Reciprocal Rank

        """
        ranks = self.get_ranks(sample_list, model_output)
        score = torch.mean(ranks.reciprocal()).float()
        return score

    def summarize(self, *args, **kwargs):
        score = np.mean(np.reciprocal(self._ranks))
        return torch.tensor(score, dtype=torch.float)
