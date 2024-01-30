# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
import torch.nn.functional as F
from antmmf.common.registry import registry
from antmmf.models.base_model import BaseModel
from antmmf.modules.layers import ConvNet

_TEMPLATES = {"number_of_answers": "{}_num_final_outputs"}

_CONSTANTS = {"hidden_state_warning": "hidden state (final) should have 1st dim as 2"}


@registry.register_model("cnn")
class CNN(BaseModel):
    """CNN is a simple model for vision tasks. CNN is supposed to act
    as a baseline to test out your stuff without any complex functionality. Passes image
    through a CNN and from a MLP to generate scores for each of the possible answers.

    Args:
        config (Configuration): Configuration node containing all of the necessary config required
                             to initialize CNN.

    Inputs: sample_list (SampleList)
        - **sample_list** should contain image attribute for image,
        targets for answer scores
    """

    def __init__(self, config):
        super().__init__(config)
        self._datasets = []
        for _, attr in registry.get("config").task_attributes.items():
            for dataset in attr.dataset_attributes:
                self._datasets.append(dataset)

    def build(self):
        assert len(self._datasets) > 0

        num_answer_choices = registry.get(
            _TEMPLATES["number_of_answers"].format(self._datasets[0])
        )

        layers_config = self.config.layers
        conv_layers = []
        for i in range(len(layers_config.input_dims)):
            conv_layers.append(
                ConvNet(
                    layers_config.input_dims[i],
                    layers_config.output_dims[i],
                    kernel_size=layers_config.kernel_sizes[i],
                )
            )
        self.cnn = nn.Sequential(*conv_layers)

        self.classifier = nn.Linear(
            self.config.classifier.input_dim, num_answer_choices
        )

    def forward(self, sample_list):
        image = sample_list.image

        image = self.cnn(image)
        if image.dim() > 1:
            image = torch.flatten(image, 1)

        scores = self.classifier(image)
        log_prob = F.log_softmax(scores, dim=-1)

        return {"logits": scores, "log_prob": log_prob}
