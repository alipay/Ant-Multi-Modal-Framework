# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.common.constants import REGISTRY_FOR_MODEL
from antmmf.common.registry import registry
from antmmf.datasets.base_dataset_builder import BaseDatasetBuilder
from .dataset import MMFRoiDataset


@registry.register_builder("roi_dataset")
class RoiDatasetBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("roi_dataset")

    def _build(self, dataset_type, config, *args, **kwargs):
        return None

    def _load(self, dataset_type, config, *args, **kwargs):
        self.dataset = MMFRoiDataset(dataset_type, config)
        return self.dataset

    def update_registry_for_model(self, config):
        super().update_registry_for_model(config)
        registry_for_model = []
        if hasattr(self.dataset, "ocr_processor"):
            processor_key = f"{self.dataset_name}.ocr_processor"
            registry.register(processor_key, self.dataset.ocr_processor)
            registry_for_model.append(processor_key)
        registry.register(REGISTRY_FOR_MODEL, registry_for_model)
