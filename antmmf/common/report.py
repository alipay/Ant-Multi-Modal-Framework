# Copyright (c) 2023 Ant Group and its affiliates.

import collections
import warnings
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch

from antmmf.common.registry import registry


class Report(OrderedDict):
    def __init__(self, batch={}, model_output: Dict = None, *args):
        super().__init__(self)
        if model_output is None:
            model_output = {}
        if self._check_and_load_tuple(batch):
            return

        all_args = [batch, model_output] + [*args]
        for idx, arg in enumerate(all_args):
            if not isinstance(arg, collections.abc.Mapping):
                raise TypeError(
                    "Argument {:d}, {} must be of instance of "
                    "collections.abc.Mapping".format(idx, arg)
                )

        self.writer = registry.get("writer")

        self.warning_string = (
            "Updating forward report with key {}"
            "{}, but it already exists in {}. "
            "Please consider using a different key, "
            "as this can cause issues during loss and "
            "metric calculations."
        )

        for idx, arg in enumerate(all_args):
            for key, item in arg.items():
                if key in self and idx >= 2:
                    log = self.warning_string.format(
                        key, "", "previous arguments to report"
                    )
                    warnings.warn(log)
                self[key] = item

    def _check_and_load_tuple(self, batch):
        if isinstance(batch, collections.abc.Mapping):
            return False

        if isinstance(batch[0], (tuple, list)) and isinstance(batch[0][0], str):
            for kv_pair in batch:
                self[kv_pair[0]] = kv_pair[1]
            return True
        else:
            return False

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        return self[key]

    def fields(self):
        return list(self.keys())


def default_result_formater(report: Report):
    if isinstance(report, collections.abc.Mapping):
        report.pop("writer", None)
        report.pop("warning_string", None)
        result = {}
        for k, v in report.items():
            result[k] = default_result_formater(v)
    elif isinstance(report, np.ndarray):
        result = report
    elif isinstance(report, torch.Tensor):
        result = report.cpu().numpy()
    elif isinstance(report, list):
        result = [default_result_formater(x) for x in report]
    else:
        result = report
    return result
