# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.common.registry import registry
from antmmf.datasets.base_dataset_builder import BaseDatasetBuilder
from .mc_qa_dataset import MMFUnivlVideoDataset_MC_QA


@registry.register_builder("video_multi_choice_qa")
class UnivlDatasetBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("video_multi_choice_qa")

    def _load(self, dataset_type, config, *args, **kwargs):
        self.dataset = MMFUnivlVideoDataset_MC_QA(dataset_type, config)
        return self.dataset
