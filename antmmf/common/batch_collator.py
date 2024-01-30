# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import traceback
from typing import Callable
from antmmf.structures import SampleList


class BatchCollator:
    # TODO: Think more if there is a better way to do this
    _IDENTICAL_VALUE_KEYS = ["dataset_type", "dataset_name"]

    def __init__(self, collate_fn: Callable = None):
        self.collate_fn = collate_fn

    def __call__(self, batch):
        if self.collate_fn is not None:
            batch = self.collate_fn(batch)
        # avoid deepcopy in case batch is already `SampleList` type
        if not isinstance(batch, SampleList):
            sample_list = SampleList(batch)
        else:
            sample_list = batch
        try:
            for key in self._IDENTICAL_VALUE_KEYS:
                sample_list[key] = sample_list[key][0]
        except Exception:
            # disable silent errors in AntMMF
            traceback.print_exc()

        return sample_list
