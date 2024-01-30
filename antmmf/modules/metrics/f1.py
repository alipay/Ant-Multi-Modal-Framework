# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from deprecated import deprecated
from sklearn.metrics import f1_score
from antmmf.common.registry import registry
from antmmf.modules.metrics.base_metric import BaseMetric
from antmmf.modules.metrics.utils import convert_to_one_hot


@registry.register_metric("f1")
class F1(BaseMetric):
    """Metric for calculating F1. Can be used with type and params
    argument for customization. params will be directly passed to sklearn
    f1 function.
    **Key:** ``f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(kwargs.pop("name", "f1"))
        self.reset()
        self._multilabel = kwargs.pop("multilabel", False)
        self._sk_kwargs = kwargs

    def reset(self):
        self._scores = torch.tensor([], dtype=torch.float64, device="cpu")
        self._expected = torch.tensor([], dtype=torch.float64, device="cpu")

    def _calculate(self, scores, expected):
        if self._multilabel:
            output = torch.sigmoid(scores)
            output = torch.round(output)
            expected = convert_to_one_hot(expected, output)
        else:
            # Multiclass, or binary case
            output = scores.argmax(dim=-1)
            if expected.dim() != 1:
                # Probably one-hot, convert back to class indices array
                expected = expected.argmax(dim=-1)

        value = f1_score(
            expected.detach().cpu(), output.detach().cpu(), **self._sk_kwargs
        )

        return expected.new_tensor(value, dtype=torch.float)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate f1 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: f1.
        """
        scores = model_output["logits"]
        expected = sample_list["targets"]

        f1_score = self._calculate(scores, expected)
        return f1_score.clone().detach()

    def collect(self, sample_list, model_output, *args, **kwargs):
        self._scores = torch.cat(
            (self._scores, model_output["logits"].detach().cpu().double())
        )
        self._expected = torch.cat(
            (self._expected, sample_list["targets"].detach().cpu().double())
        )

    def summarize(self, *args, **kwargs):
        f1_score = self._calculate(self._scores, self._expected)
        return f1_score.clone().detach()


@registry.register_metric("binary_f1")
@deprecated(
    reason="binary_f1 is deprecated, you can use f1 by changing the type from "
    "`binary_f1` to `f1` and changing the `name` in params",
    version="1.3.7",
    action="default",
)
class BinaryF1(F1):
    """Metric for calculating Binary F1.

    **Key:** ``binary_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name=kwargs.get("name", "binary_f1"))


@registry.register_metric("macro_f1")
@deprecated(
    reason="macro_f1 is deprecated, you can use f1 by changing the type from `macro_f1` "
    "to `f1` and changing the `name` and `average` in params to `macro`",
    version="1.3.7",
    action="default",
)
class MacroF1(F1):
    """Metric for calculating Macro F1 for multiclass job.

    **Key:** ``macro_f1``
    """

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", "macro_f1")
        super().__init__(average="macro", name=name, **kwargs)


@registry.register_metric("micro_f1")
@deprecated(
    reason="micro_f1 is deprecated, you can use f1 by changing the type from `micro_f1` "
    "to `f1` and changing the `name` and `average` in params to `micro`",
    version="1.3.7",
    action="default",
)
class MicroF1(F1):
    """Metric for calculating Micro F1 for multiclass job.

    **Key:** ``micro_f1``
    """

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", "micro_f1")
        super().__init__(average="micro", name=name, **kwargs)


@registry.register_metric("multilabel_f1")
@deprecated(
    reason="multilabel_f1 is deprecated, you can use f1 by changing the type from `multilabel_f1` "
    "to `f1` and setting the `multilabel` in params to `True`",
    version="1.3.7",
    action="default",
)
class MultiLabelF1(F1):
    """Metric for calculating Multilabel F1.

    **Key:** ``multilabel_f1``
    """

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", "multilabel_f1")
        super().__init__(multilabel=True, name=name, **kwargs)
