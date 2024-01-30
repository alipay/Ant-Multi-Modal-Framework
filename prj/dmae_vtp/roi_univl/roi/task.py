# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.common.registry import registry
from antmmf.tasks import BaseTask


@registry.register_task("roi_task")
class RoiTask(BaseTask):
    def __init__(self):
        super(RoiTask, self).__init__("roi_task")

    def _get_available_datasets(self):
        return ["roi_dataset"]

    def _preprocess_item(self, item):
        return item
