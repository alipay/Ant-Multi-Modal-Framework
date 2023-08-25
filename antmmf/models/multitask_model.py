import torch
import torch.nn as nn
import torchvision
from antmmf.common.registry import registry

from antmmf.models.base_model import BaseModel
from antmmf.modules.encoders.visual_encoder import VisualEncoder
from antmmf.modules.encoders.multimodal_encoder import MultimodalBertEncoder
from antmmf.modules.build import build_classifier_layer


@registry.register_model("multitask_model")
class Multitask_model(BaseModel):
    # 所有模型的初始化，需要这个config
    def __init__(self, config):
        super().__init__(config)
        self.task_names = self.config.task_names
        self.task_num = len(self.task_names)

    # 在这里定义所需要的模块
    def build(self):
        self.backbone = None
        if self.config.encoder.type in [
            "BatchEfficientNetImageEncoder",
            "BatchPVTEncoder",
        ] or hasattr(torchvision.models, self.config.encoder.type):
            self.backbone = VisualEncoder(self.config.encoder)
        elif self.config.encoder.type == "mmbt":
            self.backbone = MultimodalBertEncoder(self.config.encoder)
        else:
            print("unknow encoder type: {}\n".format(self.config.encoder.type))

        classifiers = []
        for param in self.config.classifiers:
            classifiers.append(build_classifier_layer(param))
        self.linears = nn.ModuleList(classifiers)
        if self.config.use_uncertainty_weight:
            self.uncertainty_weight = nn.Parameter(torch.zeros((self.task_num)))

    # MMF模型的forward函数的输入是一个词典，包含了有关输入特征的信息
    def forward(self, sample_list, *args, **kwargs):
        output = {}
        if self.config.encoder.type == "mmbt":
            text = sample_list["text"]  # [batch_size, text_fields*max_length]
            mask = sample_list["mask"]  # [batch_size, text_fields*max_length]

            # [batch_size, num_images, c, h, w] or [batch_size, c, h, w]
            image = sample_list["image"]

            # [batch_size, num_images]
            image_mask = sample_list.get("image_mask", None)
            # [batch_size, text_fields*max_length]
            segment = sample_list["segment"]
            cls_id = sample_list["cls_id"]  # [batch_size,]
            sep_id = sample_list["sep_id"]  # [batch_size,]
            features, rep_info, _, _ = self.backbone(
                text, mask, segment, image, cls_id, sep_id, image_mask
            )
            features = torch.flatten(features, start_dim=1)
            for i in range(self.task_num):
                output["{}_logits".format(self.task_names[i])] = self.linears[i](
                    features
                )
            output.update(rep_info)
        else:
            image = sample_list["img"]
            features = self.backbone(image)
            features = torch.flatten(features, start_dim=1)
            for i in range(self.task_num):
                output["{}_logits".format(self.task_names[i])] = self.linears[i](
                    features
                )

        if self.config.use_uncertainty_weight:
            output["uncertainty_weight"] = self.uncertainty_weight
        return output
