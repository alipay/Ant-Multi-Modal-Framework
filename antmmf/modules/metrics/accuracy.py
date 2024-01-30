# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import numpy as np
from deprecated import deprecated
from antmmf.common.registry import registry
from antmmf.modules.metrics.base_metric import BaseMetric


@registry.register_metric("accuracy")
class Accuracy(BaseMetric):
    """Metric for calculating accuracy.

    **Key:** ``accuracy``
    """

    def __init__(self, name: str = "accuracy"):
        super().__init__(name)
        self.reset()

    def reset(self):
        self._correct = 0
        self._total = 0

    def _calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: accuracy.

        """
        output = model_output["logits"]
        output = output.view(-1, output.size()[-1])
        expected = sample_list["targets"]

        # deal with mulitple instances of predictions
        output = output.view(expected.shape[0], -1, output.shape[-1]).sum(axis=1)

        assert (
            output.dim() <= 2
        ), "Output from model shouldn't have more than dim 2 for accuracy"
        assert (
            expected.dim() <= 2
        ), "Expected target shouldn't have more than dim 2 for accuracy"

        if expected.shape[0] == 0:
            return torch.tensor(0.0), 0

        if output.dim() == 2:
            output = torch.max(output, 1)[1]

        # If more than 1
        if expected.dim() == 2:
            expected = torch.max(expected, 1)[1]

        correct = (expected.to(device=output.device) == output.squeeze()).sum().float()
        total = len(expected)

        return correct, total

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: accuracy.

        """
        correct, total = self._calculate(sample_list, model_output, *args, **kwargs)

        acc = correct / max(total, 1)
        return acc.clone().detach()

    def collect(self, sample_list, model_output, *args, **kwargs):
        """
        Args:
            model_output (Dict): Dict returned by model, that contains two modalities
        Returns:
            torch.FloatTensor: Accuracy
        """
        correct, total = self._calculate(sample_list, model_output, *args, **kwargs)
        self._correct += correct.detach().cpu()
        self._total += total

    def summarize(self, *args, **kwargs):
        score = np.around((self._correct / max(self._total, 1)).numpy(), 4)
        return torch.tensor(score, dtype=torch.float)


@registry.register_metric("named_accuracy")
class NamedAccuracy(Accuracy):
    def __init__(self, *args, **kwargs):
        self.prefix = kwargs.get("prefix", None) or kwargs["name"][0]
        super().__init__(name="{}_accuracy".format(self.prefix))
        self.reset()

    def _calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: accuracy.

        """
        if f"{self.prefix}_logits" not in model_output:
            return torch.tensor(0.0), 0

        if f"{self.prefix}_targets" not in sample_list:
            return torch.tensor(0.0), 0

        output = model_output[f"{self.prefix}_logits"]
        output = output.view(-1, output.size()[-1])
        expected = sample_list[f"{self.prefix}_targets"]

        # deal with mulitple instances of predictions
        output = output.view(expected.shape[0], -1, output.shape[-1]).sum(axis=1)

        model_output = {"logits": output}
        sample_list = {"targets": expected}
        return super(NamedAccuracy, self)._calculate(
            sample_list, model_output, *args, **kwargs
        )


@registry.register_metric("node_accuracy")
class NodeAccuracy(NamedAccuracy):
    @deprecated(
        reason="node_accuracy is deprecated, you can use named_accuracy by changing the type from "
        "`node_accuracy` to `named_accuracy` and changing the `name` in params to `node`",
        version="1.3.7",
        action="default",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(name="node")
        self.reset()


# For edge classification
@registry.register_metric("edge_accuracy")
class EdgeAccuracy(NamedAccuracy):
    @deprecated(
        reason="edge_accuracy is deprecated, you can use named_accuracy by changing the type from "
        "`edge_accuracy` to `named_accuracy` and changing the `name` in params to `edge`",
        version="1.3.7",
        action="default",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(name="edge")
        self.reset()


# For link prediction
@registry.register_metric("link_accuracy")
class LinkAccuracy(NamedAccuracy):
    @deprecated(
        reason="link_accuracy is deprecated, you can use named_accuracy by changing the type from "
        "`link_accuracy` to `named_accuracy` and changing the `name` in params to `link`",
        version="1.3.7",
        action="default",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(name="link")
        self.reset()
