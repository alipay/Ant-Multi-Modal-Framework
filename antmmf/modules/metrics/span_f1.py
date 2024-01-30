# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from antmmf.common.registry import registry
from antmmf.modules.metrics.base_metric import BaseMetric
from antmmf.utils.text_utils import (
    bmeso_tag_to_spans,
    bio_tag_to_spans,
    bioes_tag_to_spans,
)


@registry.register_metric("SpanF1")
class SpanF1(BaseMetric):
    """
    Span-level F1 metric for sequence labeling tasks like NER.
    """

    def __init__(self, encoding_type, name: str = "SpanF1"):
        super().__init__(name)
        self.encoding_type = encoding_type
        self.reset()

    def reset(self):
        self._pred_true = 0
        self._pred_all = 0
        self._truth = 0

    def _calculate(self, sample_list, model_output, *args, **kwargs):
        if self.encoding_type == "bmeso":
            tag2span_func = bmeso_tag_to_spans
        elif self.encoding_type == "bio":
            tag2span_func = bio_tag_to_spans
        elif self.encoding_type == "bioes":
            tag2span_func = bioes_tag_to_spans
        else:
            raise NotImplementedError(f"{self.encoding_type} is not support.")
        output = model_output["pred"]
        output_spans = [tag2span_func(item) for item in output]
        ground_truths = [tag2span_func(item) for item in sample_list["label"]]
        pred_true = 0
        pred_all = 0
        truth = 0
        for i in range(len(output_spans)):
            output_span = output_spans[i]
            ground_truth = ground_truths[i]
            pred_all += len(output_span)
            truth += len(ground_truth)
            for item in output_span:
                if item in ground_truth:
                    pred_true += 1
        return pred_true, pred_all, truth

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate span f1 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: span f1.
        """
        pred_true, pred_all, truth = self._calculate(
            sample_list, model_output, *args, **kwargs
        )
        p = pred_true / (pred_all + 1e-20)
        r = pred_true / (truth + 1e-20)
        f1 = (2 * p * r) / (p + r + 1e-20)
        return torch.tensor(f1, dtype=torch.float)

    def collect(self, sample_list, model_output, *args, **kwargs):
        pred_true, pred_all, truth = self._calculate(
            sample_list, model_output, *args, **kwargs
        )
        self._pred_true += pred_true
        self._pred_all += pred_all
        self._truth += truth

    def summarize(self, *args, **kwargs):
        sum_p = self._pred_true / (self._pred_all + 1e-20)
        sum_r = self._pred_true / (self._truth + 1e-20)
        sum_f1 = (2 * sum_p * sum_r) / (sum_p + sum_r + 1e-20)
        return torch.tensor(sum_f1, dtype=torch.float)
