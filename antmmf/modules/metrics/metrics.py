# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

"""
The metrics module contains implementations of various metrics used commonly to
understand how well our models are performing. For e.g. accuracy, vqa_accuracy,
r@1 etc.

For implementing your own metric, you need to follow these steps:

1. Create your own metric class and inherit ``BaseMetric`` class.
2. In the ``__init__`` function of your class, make sure to call
   ``super().__init__('name')`` where 'name' is the name of your metric. If
   you require any parameters in your ``__init__`` function, you can use
   keyword arguments to represent them and metric constructor will take care of
   providing them to your class from config.
3. Implement a ``calculate`` function which takes in ``SampleList`` and
   `model_output` as input and return back a float tensor/number.
4. Register your metric with a key 'name' by using decorator,
   ``@registry.register_metric('name')``.

Example::

    import torch

    from antmmf.common.registry import registry
    from antmmf.modules.metrics import BaseMetric

    @registry.register_metric("some")
    class SomeMetric(BaseMetric):
        def __init__(self, some_param=None):
            super().__init__("some")
            ....

        def calculate(self, sample_list, model_output):
            metric = torch.tensor(2, dtype=torch.float)
            return metric

Example config for above metric::

    model_attributes:
        antmmf:
            metrics:
            - type: some
              params:
                some_param: a
"""

import collections.abc
import numpy as np
import torch
from antmmf.common.registry import registry


class Metrics:
    """Internally used by antmmf, Metrics acts as wrapper for handling
    calculation of metrics over various metrics specified by the model in
    the config. It initializes all of the metrics and when called it runs
    calculate on each of them one by one and returns back a dict with proper
    naming back. For e.g. an example dict returned by Metrics class:
    ``{'val/vqa_accuracy': 0.3, 'val/r@1': 0.8}``

    Args:
        metric_list (List[Configuration]): List of Configurations where
            specifies name and parameters of the metrics used.
    """

    def __init__(self, metric_list):
        self.dataset_name = None
        if not isinstance(metric_list, list):
            metric_list = [metric_list]

        self.writer = registry.get("writer")
        self.metrics = self._init_metrics(metric_list)

    def _init_metrics(self, metric_list):
        metrics = {}
        for metric in metric_list:
            params = {}
            if isinstance(metric, collections.abc.Mapping):
                if not hasattr(metric, "type"):
                    raise ValueError(
                        "Metric {} needs to have 'type' attribute".format(metric)
                    )
                params = getattr(metric, "params", {})
                metric = metric.type
            else:
                if not isinstance(metric, str):
                    raise TypeError(
                        "Metric {} has inappropriate type 'dict' or 'str' allowed".format(
                            metric
                        )
                    )

            metric_cls = registry.get_metric_class(metric)
            if metric_cls is None:
                raise ValueError(
                    "No metric named {} registered to registry".format(metric)
                )
            metric = metric_cls(**params)
            metrics[metric.name] = metric

        return metrics

    def __call__(self, sample_list, model_output, *args, **kwargs):
        values = {}

        if getattr(sample_list, "dataset_type", None) is None:
            # metric needs to be evaluated on a dataset_type
            # return empty if dataset_type information is not given
            return values

        dataset_type = sample_list.dataset_type

        with torch.no_grad():
            for metric_name, metric_object in self.metrics.items():
                key = "{}/{}".format(dataset_type, metric_name)
                metric_value = metric_object._calculate_with_checks(
                    sample_list, model_output, *args, **kwargs
                )
                if metric_value is None:
                    # this can gracely handle cases where there is no metric
                    # computed
                    continue

                if isinstance(metric_value, dict):
                    # support returning a series of metrics
                    for name, val in metric_value.items():
                        key = "{}/{}".format(dataset_type, name)
                        values[key] = val
                else:
                    # metric is a single value
                    values[key] = metric_value

            for metric_key in values.keys():
                if not isinstance(values[metric_key], torch.Tensor):
                    values[metric_key] = torch.tensor(
                        values[metric_key], dtype=torch.float
                    )
                else:
                    values[metric_key] = values[metric_key].float()

                if values[metric_key].dim() == 0:
                    values[metric_key] = values[metric_key].view(1)

        registry.register(
            "{}.{}.{}".format("metrics", sample_list.dataset_name, dataset_type), values
        )

        return values

    def reset(self):
        for _, metric_object in self.metrics.items():
            metric_object.reset()

    def collect(self, sample_list, model_output, *args, **kwargs):
        self.dataset_type = sample_list.dataset_type
        self.dataset_name = sample_list.dataset_name
        with torch.no_grad():
            for metric_name, metric_object in self.metrics.items():
                metric_object.collect(sample_list, model_output, *args, **kwargs)

    def summarize(self, *args, **kwargs):
        values = {}
        with torch.no_grad():
            for metric_name, metric_object in self.metrics.items():
                key = "{}/{}".format(self.dataset_type, metric_name)
                score = metric_object.summarize(*args, **kwargs)
                if isinstance(score, dict):
                    for k, v in score.items():
                        # follow same naming convention of metric
                        lcl_key = "{}/{}".format(self.dataset_type, k)
                        values[lcl_key] = np.around(v, 4)
                else:
                    values[key] = np.around(score, 4)
        summary = values
        registry.register(
            "{}.{}.{}".format("overall metrics", self.dataset_name, self.dataset_type),
            summary,
        )
        return summary
