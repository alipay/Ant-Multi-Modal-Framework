# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from antmmf.common.registry import registry
from antmmf.modules.metrics.f1 import F1


@registry.register_metric("hier_multilabel_f1")
class HierMultilabelF1(F1):
    """ """

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", "hier_multilabel_f1")
        super().__init__(multilabel=True, average="micro", name=name, **kwargs)
        self.reset()
        kwargs.pop("hier_label_schema", False)
        self._sk_kwargs.pop("hier_label_schema", False)

    def get_scores_and_expected(self, sample_list, model_output):
        scores = torch.tensor([], dtype=torch.float64, device="cpu")
        expected = torch.tensor([], dtype=torch.float64, device="cpu")
        hier_labels = sample_list["hier_label"]
        hier_label_nums = sample_list["hier_label_num"]
        hier_logits = model_output["hier_logits"]
        nbz = hier_labels.size(0)
        for batch_idx in range(nbz):
            """Calculate accuracy & recall."""
            logits = hier_logits[0][batch_idx]
            hier_label_padding = hier_labels[batch_idx][0]
            hier_label = hier_label_padding[: hier_label_nums[batch_idx]]
            expected_tmp = (
                torch.zeros_like(logits)
                .scatter_(0, hier_label.long(), 1)
                .long()
                .unsqueeze(0)
            )
            scores = torch.cat((scores, logits.detach().cpu().double().unsqueeze(0)), 0)
            expected = torch.cat((expected, expected_tmp.detach().cpu().double()), 0)
        return scores, expected

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate f1 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: f1.
        """
        scores, expected = self.get_scores_and_expected(sample_list, model_output)
        f1_score = self._calculate(scores, expected)
        return torch.tensor(f1_score.item(), dtype=torch.float)

    def collect(self, sample_list, model_output, *args, **kwargs):
        scores, expected = self.get_scores_and_expected(sample_list, model_output)
        self._scores = torch.cat((self._scores, scores))
        self._expected = torch.cat((self._expected, expected))
