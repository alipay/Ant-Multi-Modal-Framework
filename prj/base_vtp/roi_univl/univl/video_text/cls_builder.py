# coding: utf-8
# Copyright (c) Ant Group. All rights reserved.

from antmmf.common.registry import registry
from antmmf.datasets.base_dataset_builder import BaseDatasetBuilder
from .cls_dataset import MMFVideoClassificationDataset


@registry.register_builder("video_text_classification")
class UnivlDatasetBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("video_text_classification")

    def _load(self, dataset_type, config, *args, **kwargs):
        self.dataset = MMFVideoClassificationDataset(dataset_type, config)
        return self.dataset
