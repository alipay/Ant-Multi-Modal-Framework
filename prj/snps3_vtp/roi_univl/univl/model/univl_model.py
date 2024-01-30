# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import torch

from antmmf.common.registry import registry
from antmmf.models.base_model import BaseModel
from .univl_classification import UnivlBertForClassification
from .univl_pretrain import UnivlForPretraining
from .univl_video_cls import UnivlForVideoClassification
from .univl_video_pretrain import UnivlForVideoPretraining
from .univl_video_ret import UnivlForVideoTextRetrieval
from .univl_video_multi_choice_qa import UnivlForVideoMultiChoiceQA


@registry.register_model("univl")
class Univl(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        if self.config.training_head_type == "pretraining":
            self.model = UnivlForPretraining(self.config)
        elif self.config.training_head_type == "classification":
            self.model = UnivlBertForClassification(self.config)
        elif self.config.training_head_type == "video_text_classification":
            self.model = UnivlForVideoClassification(self.config)
        elif self.config.training_head_type == "video_text_retrieval":
            self.model = UnivlForVideoTextRetrieval(self.config)
            # extract modality features for retrival evaluation
            self.get_l2_input = self.model.module.get_l2_input
        elif self.config.training_head_type == "video_multi_choice_qa":
            self.model = UnivlForVideoMultiChoiceQA(self.config)
        elif self.config.training_head_type == "video_pretraining":
            self.model = UnivlForVideoPretraining(self.config)

    def group_inputs(self, sample_list):
        key_input_group = {
            "ocr": None,
            "caption": None,
            "region": None,
            "image": None,
            "generation": None,
        }
        for sample_key in sample_list.keys():
            for input_key in key_input_group.keys():
                if sample_key.startswith(input_key):
                    if key_input_group[input_key] is None:
                        key_input_group[input_key] = {}
                    key_input_group[input_key][sample_key] = sample_list[sample_key]
        return key_input_group

    def forward(self, sample_list, *args, **kwargs):
        key_input_pairs = self.group_inputs(sample_list)

        model_output = self.model(
            key_input_pairs["image"],
            key_input_pairs["caption"],
            key_input_pairs["ocr"],
            key_input_pairs["region"],
            caption_output=key_input_pairs["generation"],
            sample_list=sample_list,
        )

        if "pretraining" in self.config.training_head_type:
            output = {"losses": {}, "metrics": {}}
            for head_output in model_output:
                output["losses"].update(head_output["losses"])
                output["metrics"].update(head_output.get("metrics", {}))

        else:
            output = (
                {"logits": model_output}
                if isinstance(model_output, (torch.Tensor,))
                else model_output
            )

        return output

    def _default_optimizer_parameters(self, config):
        lr = config.optimizer_attributes.params.lr
        weight_decay = config.optimizer_attributes.params.weight_decay

        img_encoder = ["img_encoder"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in img_encoder)
                ],
                "lr": lr * 0.1,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in img_encoder)
                ],
                "lr": lr,
                "weight_decay": weight_decay,
            },
        ]
        return optimizer_grouped_parameters

    def get_optimizer_parameters(self, config):
        if hasattr(self.model, "get_optimizer_parameters"):
            return self.model.get_optimizer_parameters(config)
        else:
            return self._default_optimizer_parameters(config)
