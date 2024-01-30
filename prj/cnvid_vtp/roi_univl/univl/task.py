# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.common.registry import registry
from antmmf.tasks import BaseTask


@registry.register_task("univl_task")
class UnivlTask(BaseTask):
    def __init__(self):
        super(UnivlTask, self).__init__("univl_task")

    def _get_available_datasets(self):
        return [
            "univl_dataset",
            "video_text_pretrain",
            "video_text_retrieval",
            "video_multi_choice_qa",
            "video_text_classification",
        ]

    def _preprocess_item(self, item):
        return item
