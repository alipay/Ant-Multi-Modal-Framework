# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from antmmf.common.registry import registry
from antmmf.modules.metrics.accuracy import Accuracy
from antmmf.modules.utils import build_hier_tree


@registry.register_metric("hier_label_accuracy")
class HierLabelAccuracy(Accuracy):
    """Metric for calculating hier_label_accuracy.
    In hierarchical softmax, the parent node' probability will always
    larger than its children's. So output both parent and its children
    probabilities together and perform argmax label prediction will never
    get right predictions of its children. Here we employ hier comparasion
    to evaluate.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            strict: 为False时. 对于target label节点 和 pred label节点 相同 或 target label 是 pred label的上位节点时，
                视为预测正确. 比如 target label为 时尚 , pred 为 时尚-新技术汽车 or 时尚 则预测正确，
                target label为 时尚-新技术汽车, pred label为 时尚 则预测错误.
                    为True时. 标准更为严格，只对于target label节点 和 pred label节点 相同视为预测正确.
                比如 target label为 时尚 , pred 为 时尚-新技术汽车 则预测错误.
        """
        super(HierLabelAccuracy, self).__init__(
            name=kwargs.get("name", "hier_label_accuracy")
        )
        hier_label_schema = kwargs["hier_label_schema"]
        self.tree = build_hier_tree(hier_label_schema)
        self.strict = kwargs["strict"] if "strict" in kwargs else False

    def _calculate(self, sample_list, model_output, *args, **kwargs):
        """
        计算层级label的hier_label_accuracy metric.
        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: correct, total
        """
        inference_result = model_output["pred_hier_tags"]
        target_hier_tags = sample_list["target_hier_tags"]

        correct, total = torch.tensor(0.0), torch.tensor(0.0)
        for sample_idx, sample_res in enumerate(inference_result):
            total += 1.0
            target_hier_tag = target_hier_tags[sample_idx]
            pred_hier_tag = sample_res["result"]["label"]
            if self.strict and pred_hier_tag == target_hier_tag:
                correct += 1.0
            elif not self.strict and self.tree.compare_hier_label(
                pred_hier_tag, target_hier_tag
            ):
                correct += 1.0
        return correct, total
