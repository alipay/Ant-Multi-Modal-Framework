# Copyright (c) 2023 Ant Group and its affiliates.
import torch
from antmmf.common.registry import registry
from antmmf.models.base_model import BaseModel
from antmmf.modules.decoders import HierarchicalClassifier
from antmmf.modules.encoders import VisualEncoder
from antmmf.modules.build import build_classifier_layer


@registry.register_model("image_model")
class ImageBackbone(BaseModel):
    # 所有模型的初始化，需要这个config
    # 这个config应该包含所有模型需要用到的参数设置
    def __init__(self, config):
        super().__init__(config)
        self._update_image_encoder = config.get("update_image_encoder", True)
        self._update_classifier = config.get("update_classifier", True)
        self._datasets = []
        for _, attr in registry.get("config").task_attributes.items():
            for dataset in attr.dataset_attributes:
                self._datasets.append(dataset)

    # 在这里定义所需要的模块
    def build(self):

        self.image_module = VisualEncoder(self.config.image_encoder).module
        self.do_hir_cls = self.config.get("hier_label_schema", None)
        if self.do_hir_cls:
            self.hir_classifier = HierarchicalClassifier(
                self.image_module.out_dim, self.config.hier_label_schema
            )
        else:
            self.classifier = build_classifier_layer(self.config.classifier)

    def train(self, mode=True):
        """
        Override the default train() to allow freezing image encoder, text encoder, and classifier
        :return:
        """
        super().train(mode)
        count = 0
        if mode:
            if self._update_image_encoder is False:
                self.writer.write("Freezing image encoder.")
                for m in self.image_module.modules():
                    m.eval()
                    m.requires_grad = False
            if self._update_classifier is False:
                self.writer.write("Freezing classifier.")
                for m in self.classifier.modules():
                    m.eval()
                    m.requires_grad = False

    # MMF模型的forward函数的输入是一个词典，包含了有关输入特征的信息
    def forward(self, sample_list, *args, **kwargs):
        image = sample_list["img"]
        image_features = self.image_module(image)
        image_features = torch.flatten(image_features, start_dim=1)

        if self.do_hir_cls:
            output = self.hir_classifier(image_features)
        else:
            logits = self.classifier(image_features)
            output = {"logits": logits}

        return output
