# Copyright (c) 2023 Ant Group and its affiliates.

import numpy as np
import torch.nn.functional as F

from antmmf.common.registry import registry
from antmmf.predictors.base_predictor import BasePredictor


@registry.register_predictor("MultitaskPredictor")
class MultitaskPredictor(BasePredictor):
    def __init__(self, config):
        super(MultitaskPredictor, self).__init__(config)
        task_predictor = registry.get_predictor_class(
            self.config.predictor_parameters.task_predictor_name
        )
        self.task_predictor = task_predictor(config)
        self.task_predictor.load()

        self.task_names = self.config.predictor_parameters.task_names
        self.task_names_map = None
        self.label_list_map = None

        if (
            getattr(self.config.predictor_parameters, "task_names_map", None)
        ) is not None:
            self.task_names_map = self.config.predictor_parameters.task_names_map
        if (
            getattr(self.config.predictor_parameters, "label_list_map", None)
        ) is not None:
            self.label_list_map = self.config.predictor_parameters.label_list_map

    def dummy_request(self):
        return self.task_predictor.dummy_request()

    def _build_sample(self, data, json_obj):
        return self.task_predictor._build_sample(data, json_obj)

    def format_result(self, report):
        predicts = {}
        for key in report.keys():
            if "logits" not in key:
                continue
            task_name = key.split("_logits")[0]
            if self.task_names_map is not None:
                task_name = self.task_names_map[self.task_names.index(task_name)]

            probs = F.softmax(report[key], -1)[0]
            labels = probs.argmax(-1)
            prob = np.around(probs[labels].cpu().numpy(), decimals=4).item()
            label = np.around(labels.cpu().numpy(), decimals=4).item()
            if self.label_list_map is not None:
                label = self.label_list_map[task_name][label]

            predicts[task_name] = {
                "prob": prob,
                "label": label,
                "logits": report[key].tolist(),
            }
        return predicts
