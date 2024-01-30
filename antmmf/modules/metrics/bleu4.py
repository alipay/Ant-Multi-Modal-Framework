# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

from antmmf.utils.text_utils import EOS_INDEX, PAD_INDEX, SOS_INDEX, keep_till_eos
from antmmf.common.registry import registry
from antmmf.modules.metrics.base_metric import BaseMetric


@registry.register_metric("bleu4")
class Bleu4Metric(BaseMetric):
    """Metric for calculating BLEU4 Score."""

    def __init__(self, name: str = "bleu4"):
        super().__init__(name)
        self.ignore_indices = set()
        self.ignore_indices.add(PAD_INDEX)
        self.ignore_indices.add(SOS_INDEX)
        self.ignore_indices.add(EOS_INDEX)
        self.reset()

    def reset(self):
        self.bleu = [[] for i in range(4)]

    def collect(self, sample_list, model_output, *args, **kwargs):
        bleu = self.calculate(sample_list, model_output)
        for i, s in enumerate(bleu.values()):
            self.bleu[i].append(s)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            dict: bleu score of 1-4 grams.

        """

        references = []

        # References
        for j, p in enumerate(sample_list.answers):
            ref = keep_till_eos(p.tolist())
            img_captions = [str(c) for c in ref if c not in self.ignore_indices]
            references.append([img_captions])

        # Hypotheses
        scores = torch.max(model_output["logits"], dim=-1)[1]
        hypotheses = []
        for j, p in enumerate(scores):
            hyp = keep_till_eos(p.tolist())
            hypotheses.append([str(c) for c in hyp if c not in self.ignore_indices])

        assert len(references) == len(hypotheses)
        all_bleu = []
        for weight in [
            [1.0],
            [0.5, 0.5],
            [0.333, 0.333, 0.334],
            [0.25, 0.25, 0.25, 0.25],
        ]:
            all_bleu.append(corpus_bleu(references, hypotheses, weights=weight))

        return {
            "bleu1": all_bleu[0],
            "bleu2": all_bleu[1],
            "bleu3": all_bleu[2],
            "bleu4": all_bleu[3],
        }

    def summarize(self, *args, **kwargs):
        all_bleu = [np.mean(s) for s in self.bleu]
        return {
            "bleu1": all_bleu[0],
            "bleu2": all_bleu[1],
            "bleu3": all_bleu[2],
            "bleu4": all_bleu[3],
        }
