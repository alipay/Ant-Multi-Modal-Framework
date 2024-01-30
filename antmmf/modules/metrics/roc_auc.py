# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from deprecated import deprecated
from antmmf.common.registry import registry
from antmmf.modules.metrics.base_metric import BaseMetric
from antmmf.modules.metrics.utils import convert_to_one_hot
from sklearn.metrics import roc_auc_score


@registry.register_metric("roc_auc")
class ROC_AUC(BaseMetric):
    """Metric for calculating ROC_AUC.
    See more details at `sklearn.metrics.roc_auc_score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_ # noqa

    **Note**: ROC_AUC is not defined when expected tensor only contains one
    label. Make sure you have both labels always or use it on full val only

    **Key:** ``roc_auc``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name=kwargs.get("name", "roc_auc"))
        self._sk_kwargs = kwargs
        self.reset()

    def reset(self):
        self._output = torch.tensor([], dtype=torch.float64, device="cpu")
        self._expected = torch.tensor([], dtype=torch.float64, device="cpu")

    def _calculate(self, output, expected):
        output = torch.nn.functional.softmax(output, dim=-1)
        expected = convert_to_one_hot(expected, output)
        value = roc_auc_score(expected.cpu(), output.cpu(), **self._sk_kwargs)
        return torch.tensor(value, dtype=torch.float)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate ROC_AUC and returns it back. The function performs softmax
        on the logits provided and then calculated the ROC_AUC.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "logits"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: ROC_AUC.

        """

        return self._calculate(model_output["logits"], sample_list["targets"])

    def collect(self, sample_list, model_output, *args, **kwargs):
        self._output = torch.cat((self._output, model_output["logits"].detach().cpu()))
        self._expected = torch.cat(
            (self._expected, sample_list["targets"].detach().cpu())
        )

    def summarize(self, *args, **kwargs):
        return self._calculate(self._output, self._expected)


@registry.register_metric("micro_roc_auc")
@deprecated(
    reason="micro_roc_auc is deprecated, you can use roc_auc by changing the type from "
    "`micro_roc_auc` to `roc_auc` and changing the `average` in params to `micro`",
    version="1.3.7",
    action="default",
)
class MicroROC_AUC(ROC_AUC):
    """Metric for calculating Micro ROC_AUC.

    **Key:** ``micro_roc_auc``
    """

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", "micro_roc_auc")
        super().__init__(average="micro", name=name, **kwargs)


@registry.register_metric("macro_roc_auc")
@deprecated(
    reason="macro_roc_auc is deprecated, you can use roc_auc by changing the type from "
    "`macro_roc_auc` to `roc_auc` and changing the `average` in params to `macro`",
    version="1.3.7",
    action="default",
)
class MacroROC_AUC(ROC_AUC):
    """Metric for calculating Macro ROC_AUC.

    **Key:** ``macro_roc_auc``
    """

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", "macro_roc_auc")
        super().__init__(average="macro", name=name, **kwargs)
