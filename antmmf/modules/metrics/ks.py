# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from antmmf.common.registry import registry
from antmmf.modules.metrics.base_metric import BaseMetric
from sklearn.metrics import roc_curve


@registry.register_metric("ks")
class KS(BaseMetric):
    """Metric for calculating Kolmogorov–Smirnov statistic.
    KS is the maximum difference between the cumulative true positive and cumulative false positive rate.
    The code below calculates this using the ROC curve.
    https://cran.r-project.org/doc/contrib/Sharma-CreditScoring.pdf
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name=kwargs.get("name", "ks"))
        self.reset()

    def reset(self):
        self._output = []
        self._expected = []

    def _calculate(self, output, expected):
        fpr, tpr, _ = roc_curve(expected, output)
        return torch.tensor(max(tpr - fpr), dtype=torch.float)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate KS score

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "logits"
                                 field pointing to scores returned from the model.

        Returns:
            torch.FloatTensor: KS score.

        """

        output = model_output["logits"].view(-1)
        expected = sample_list["targets"]
        score = self._calculate(output.detach().cpu(), expected.detach().cpu())
        return score

    def collect(self, sample_list, model_output, *args, **kwargs):
        self._output.extend(model_output["logits"].view(-1).tolist())
        self._expected.extend(sample_list["targets"].tolist())

    def summarize(self, *args, **kwargs):
        score = self._calculate(self._output, self._expected)
        return score
