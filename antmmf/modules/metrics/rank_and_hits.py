# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from antmmf.common.registry import registry
from antmmf.modules.metrics.recall_at_k import RecallAtK, get_gt_ranks, score_to_ranks


@registry.register_metric("rank_and_hits")
class RankAndHitsMetric(RecallAtK):
    def __init__(self, name: str = "rank_and_hits"):
        super().__init__(name=name)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        predicted = model_output["logits"].cpu()
        batch_size = predicted.size(0)
        num_entites = predicted.size(1)

        # all_entity2 = (batch_size, max_number_positive_entities) with ids
        #       of all known positive entities in all splits
        # entity2 = (batch_size, ) of the target entity2 id
        all_entity2 = sample_list["targets"].cpu()
        entity2 = sample_list["e2"]
        entity2 = entity2.cpu()

        # zero out the actual entities EXCEPT for the target e2 entity
        for i in range(batch_size):
            e2_idx = entity2[i].item()
            e2_p = predicted[i][e2_idx].item()
            predicted[i][all_entity2[i]] = 0
            predicted[i][e2_idx] = e2_p

        ranks = score_to_ranks(predicted)
        gt_ranks = get_gt_ranks(ranks, entity2, ans_is_index=True)
        return torch.mean(gt_ranks.float().reciprocal())
