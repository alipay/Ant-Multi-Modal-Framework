# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import numpy as np
from antmmf.common.registry import registry
from antmmf.modules.metrics.base_metric import BaseMetric
from antmmf.modules.metrics.accuracy import Accuracy


@registry.register_metric("multi_accuracy")
class MultiAccuracy(BaseMetric):
    """Metric for calculating multitask accuracy.

    **Key:** ``multi_accuracy``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name=kwargs.get("name", "multi_accuracy"))
        self.task_names = kwargs["task_names"]
        self.task_acc_inst_map = {}

        for task_name in self.task_names:
            self.task_acc_inst_map[task_name] = Accuracy()

        self.reset()

    def reset(self):
        for task_name in self.task_names:
            self.task_acc_inst_map[task_name].reset()

    def _ignore_sample(self, sample_list, model_output, task, ignored_label=-1):
        output_list = model_output["{}_logits".format(task)].tolist()
        expect_list = sample_list["{}_targets".format(task)].tolist()
        expect_list_new = []
        output_list_new = []

        for i in range(len(expect_list)):
            if expect_list[i] != ignored_label:
                expect_list_new.append(expect_list[i])
                output_list_new.append(output_list[i])
        sample_list["targets"] = torch.Tensor(expect_list_new)
        model_output["logits"] = torch.Tensor(output_list_new)

        return sample_list, model_output

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: accuracy.

        """
        acc_ret = {}
        batch_correct = 0
        batch_total = 0
        for task in self.task_names:
            sample_list, model_output = self._ignore_sample(
                sample_list, model_output, task
            )
            score = torch.tensor(0)
            if min(sample_list["targets"].shape) or min(model_output["logits"].shape):
                correct, total = self.task_acc_inst_map[task]._calculate(
                    sample_list, model_output, task, *args, **kwargs
                )
                score = correct / (total + 1e-20)
                batch_correct += correct
                batch_total += total
            acc_ret["{}_accuracy".format(task)] = score.clone().detach().cpu()

        acc_ret["batch_accuracy"] = (batch_correct / batch_total).clone().detach().cpu()

        return acc_ret

    def collect(self, sample_list, model_output, *args, **kwargs):
        for task in self.task_names:
            self._ignore_sample(sample_list, model_output, task)
            if min(sample_list["targets"].shape) or min(model_output["logits"].shape):
                self.task_acc_inst_map[task].collect(sample_list, model_output)

    def summarize(self, *args, **kwargs):
        acc_total = {}
        total_correct = 0
        total = 0
        for task in self.task_names:
            acc_total["{}_accuracy".format(task)] = self.task_acc_inst_map[
                task
            ].summarize()
            total_correct += self.task_acc_inst_map[task]._correct
            total += self.task_acc_inst_map[task]._total
        total_score = np.around((total_correct / (total + 1e-20)).numpy(), 4)
        acc_total["total_accuracy"] = torch.tensor(
            total_score, dtype=torch.float, device="cpu"
        )
        return acc_total
