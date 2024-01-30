# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from antmmf.common.registry import registry
from antmmf.modules.metrics.base_metric import BaseMetric
from antmmf.modules.metrics.f1 import F1


@registry.register_metric("multi_macro_f1")
class MultiMacroF1(BaseMetric):
    """Metric for calculating F1. Can be used with type and params
    argument for customization. params will be directly passed to sklearn
    f1 function.
    **Key:** ``multi_macro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name=kwargs.get("name", "multi_macro_f1"))
        self.task_names = kwargs["task_names"]
        self.task_macro_f1_inst_map = {}

        for task_name in self.task_names:
            self.task_macro_f1_inst_map[task_name] = F1(
                average="macro", name="macro_f1"
            )
        self.reset()

    def reset(self):
        for task in self.task_names:
            self.task_macro_f1_inst_map[task].reset()

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
        """Calculate multi macro f1 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: multi marco f1.
        """
        f1_ret = {}
        for task in self.task_names:
            sample_list, model_output = self._ignore_sample(
                sample_list, model_output, task
            )
            f1_score = torch.tensor(0)
            if min(sample_list["targets"].shape) or min(model_output["logits"].shape):
                f1_score = self.task_macro_f1_inst_map[task].calculate(
                    sample_list, model_output
                )
            f1_ret["{}_macro_f1".format(task)] = f1_score.clone().detach().cpu()

        return f1_ret

    def collect(self, sample_list, model_output, *args, **kwargs):
        for task in self.task_names:
            sample_list, model_output = self._ignore_sample(
                sample_list, model_output, task
            )
            if min(sample_list["targets"].shape) or min(model_output["logits"].shape):
                self.task_macro_f1_inst_map[task].collect(sample_list, model_output)

    def summarize(self, *args, **kwargs):
        f1_score = {}
        for task in self.task_names:
            f1_score_task = self.task_macro_f1_inst_map[task].summarize()
            f1_score["{}_macro_f1".format(task)] = f1_score_task.clone().detach()
        return f1_score
