# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import numpy as np
from rouge import Rouge
from antmmf.common.registry import registry
from antmmf.modules.metrics import BaseMetric
from antmmf.utils.text_utils import EOS_INDEX, PAD_INDEX, SOS_INDEX, keep_till_eos


@registry.register_metric("rouge")
class RougeMetric(BaseMetric):
    """Metric for calculating Rouge Score."""

    def __init__(self, name: str = "rouge"):
        super().__init__(name)
        self.ignore_indices = set()
        self.ignore_indices.add(PAD_INDEX)
        self.ignore_indices.add(SOS_INDEX)
        self.ignore_indices.add(EOS_INDEX)
        self.reset()
        self.rouge = Rouge()

    def reset(self):
        self.rouge_score = {}
        for k in ["rouge-1", "rouge-2", "rouge-l"]:
            for v_k in ["p", "r", "f"]:
                self.rouge_score[k + "_" + v_k] = []

    def collect(self, sample_list, model_output, *args, **kwargs):
        rouge_score = self.calculate(sample_list, model_output)
        for k, v in rouge_score.items():
            self.rouge_score[k].append(v)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate rouge score and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            dict: rouge score of 1-4 grams.

        """

        references = []

        # References
        for j, p in enumerate(sample_list.answers):
            ref = keep_till_eos(p.tolist())
            ref = [str(c) for c in ref if c not in self.ignore_indices]
            references.append(" ".join(ref))

        # Hypotheses
        scores = torch.max(model_output["logits"], dim=-1)[1]
        hypotheses = []
        for j, p in enumerate(scores):
            hyp = keep_till_eos(p.tolist())
            hyp = [str(c) for c in hyp if c not in self.ignore_indices]
            if len(hyp) == 0:
                hyp = ["0"]
            hypotheses.append(" ".join(hyp))

        assert len(references) == len(hypotheses)
        rouge_score_ = self.rouge.get_scores(hypotheses, references, avg=True)
        rouge_score = {}
        for k, v in rouge_score_.items():
            for v_key, v_val in v.items():
                rouge_score[k + "_" + v_key] = v_val
        return rouge_score

    def summarize(self, *args, **kwargs):
        rouge_score = {}
        for k, v in self.rouge_score.items():
            rouge_score[k] = np.mean(v)
        return rouge_score
