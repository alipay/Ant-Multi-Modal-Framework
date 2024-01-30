# Copyright (c) 2023 Ant Group and its affiliates.
import torch
from torch.nn import functional as F

# 用来读取注册的keys
from antmmf.common.registry import registry

# 需要继承的模型的基本类
from antmmf.models.base_model import BaseModel

# Builder methods for image encoder and classifier
from antmmf.modules.build import build_classifier_layer
from antmmf.modules.encoders import VisualEncoder, TextEncoder

"""
注册这个新的模型
"""


@registry.register_model("concat_bert")
class ConcatBERT(BaseModel):
    # 所有模型的初始化，需要这个config
    # 这个config应该包含所有模型需要用到的参数设置
    def __init__(self, config):
        super().__init__(config)
        self._update_text_encoder = config.get("update_text_encoder", True)
        self._update_image_encoder = config.get("update_image_encoder", True)
        self._update_classifier = config.get("update_classifier", True)
        self._datasets = []
        for _, attr in registry.get("config").task_attributes.items():
            for dataset in attr.dataset_attributes:
                self._datasets.append(dataset)

    # 在这里定义所需要的模块
    def build(self):
        """
        config中的image_encoder所含参数如下：

        # "type" 参数告知所采用的encoder类型
        #   这里用 type: resnet152
        # 参数在params:下
        params:
            # 是否用公开的预训练模型
            pretrained: true
            # Pooling 类别, 如果用AdaptiveMaxPool2D则设为max
            pool_type: avg
            # 输出特征的数目，-1表示采用resnet模型中的缺省数目
            # 除此外，可以设置的参数从1到9
            num_output_features: 1
        """
        self.image_module = VisualEncoder(self.config.image_encoder).module

        """
        文本的模型设置如下
        # 类别
        type: transformer
        # 参数
        params:
            # BERT model type
            bert_model_name: bert-base-uncased
            hidden_size: 768
            # Number of BERT layers
            num_hidden_layers: 12
            # Number of attention heads in the BERT layers
            num_attention_heads: 12
        """
        self.language_module = TextEncoder(self.config.text_encoder).module

        """
        分类器设置许下
        # 类别，这里是mlp
        type: mlp
        # 参数
        params:
            # 分类器的输入维度
            # 视觉特征+文本特征 : 2048 + 768
            in_dim: 2816
            # 分类器的输出维度
            out_dim: 2
            # 多少层MLP
            num_layers: 2
        """
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
            if self._update_text_encoder is False:
                self.writer.write("Freezing text encoder.")
                for m in self.language_module.modules():
                    m.eval()
                    m.requires_grad = False
            if self._update_classifier is False:
                self.writer.write("Freezing classifier.")
                for m in self.classifier.modules():
                    m.eval()
                    m.requires_grad = False

    # MMF模型的forward函数的输入是一个词典，包含了有关输入特征的信息
    def forward(self, sample_list):
        # 注意：这些key应该和后面讲到的特征提取相对应
        # 文本特征可以依据text作为key来获得
        text = sample_list["text"]
        # 图像数据可以依据image作为key来获得
        image = sample_list["image"]

        attention_mask = sample_list["mask"]

        # 或者文本与图像特征
        text_features = self.language_module(text, attention_mask=attention_mask)[
            1
        ]  # 这个[1]是因为
        image_features = self.image_module(image)

        # 定义B为batch size，
        # 如下将图像特征的视角变为[B,-1]
        image_features = torch.flatten(image_features, start_dim=1)
        # 如下将文本特征的视角变为[B,-1]
        text_features = torch.flatten(text_features, start_dim=1)

        # 将多个模态的特征相连接
        # 维度将为[B, -1]
        combined = torch.cat([text_features, image_features], dim=1)

        # 通过分类器来输出
        logits = self.classifier(combined)

        # 将返回的logits作为value放到key="logits"
        # 注意：logits是必须的
        output = {"logits": logits}
        if "image_name" in sample_list:
            output.update({"image_name": sample_list.image_name})

        # 返回
        return output


@registry.register_model("openai_clip")
class CLIP(BaseModel):
    # 所有模型的初始化，需要这个config
    # 这个config应该包含所有模型需要用到的参数设置
    def __init__(self, config):
        super().__init__(config)
        self._update_text_encoder = config.get("update_text_encoder", True)
        self._update_image_encoder = config.get("update_image_encoder", True)
        self._loss_type = config.get("loss_type", "contrastive_loss")
        self._datasets = []
        for _, attr in registry.get("config").task_attributes.items():
            for dataset in attr.dataset_attributes:
                self._datasets.append(dataset)

    # 在这里定义所需要的模块
    def build(self):

        self.image_module = VisualEncoder(self.config.image_encoder).module

        self.language_module = TextEncoder(self.config.text_encoder).module

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
            if self._update_text_encoder is False:
                self.writer.write("Freezing text encoder.")
                for m in self.language_module.modules():
                    m.eval()
                    m.requires_grad = False

    @staticmethod
    def cosine_sim(img_embed, text_embed):
        """Cosine similarity between all the image and sentence pairs"""
        # norm
        img_embed = img_embed / torch.norm(img_embed, dim=1, keepdim=True)
        text_embed = text_embed / torch.norm(text_embed, dim=1, keepdim=True)
        return img_embed.mm(text_embed.t())

    @staticmethod
    def contrastive_loss(img_embed, text_embed, gamma=0.1):

        targets = torch.diag(torch.ones(img_embed.size(0)))
        targets = targets.to(device=img_embed.device)

        def cross_entropy(logits, targets, dim=-1):
            loss = -targets * F.log_softmax(logits, dim=dim)
            return loss.sum(dim=dim)

        # compute image-sentence score matrix
        img2txt_sim = CLIP.cosine_sim(img_embed, text_embed) / gamma
        image_loss = cross_entropy(img2txt_sim, targets, dim=1)
        text_loss = cross_entropy(img2txt_sim, targets, dim=0)
        loss = (text_loss + image_loss) / 2.0
        return loss.mean()

    @staticmethod
    def circle_loss(img_embed, text_embed, m=0.45, gamma=32):
        batch_size = img_embed.size(0)
        scores = CLIP.cosine_sim(img_embed, text_embed)
        sp_index = torch.diag(torch.ones(batch_size))
        sp = torch.reshape(scores[sp_index == 1], [batch_size, 1])
        sn = torch.reshape(scores[sp_index == 0], [batch_size, batch_size - 1])
        ap = torch.clamp_min(-sp.detach() + 1.0 + m, min=1e-12)
        an = torch.clamp_min(sn.detach() + m, min=1e-12)
        delta_p = 1.0 - m
        delta_n = m
        logit_p = torch.exp(-ap * (sp - delta_p) * gamma)
        logit_n = torch.exp(an * (sn - delta_n) * gamma)
        pos_loss = torch.sum(logit_p, dim=1)
        neg_loss = torch.sum(logit_n, dim=1)
        loss = torch.mean(torch.log(1.0 + neg_loss * pos_loss))
        return loss

    def forward(self, sample_list):

        text = sample_list["text"]
        image = sample_list["image"]

        text_features = self.language_module(text)
        image_features = self.image_module(image)

        image_features = torch.flatten(image_features, start_dim=1)
        text_features = torch.flatten(text_features, start_dim=1)

        if self._loss_type == "contrastive_loss":
            loss = CLIP.contrastive_loss(image_features, text_features)
        else:
            loss = CLIP.circle_loss(image_features, text_features)
        # acc
        similarity = CLIP.cosine_sim(text_features, image_features)
        y = torch.arange(len(similarity)).to(device=similarity.device)
        img2txt_match_idx = similarity.argmax(dim=0)
        txt2img_match_idx = similarity.argmax(dim=1)
        img_acc = (img2txt_match_idx == y).float().mean()
        txt_acc = (txt2img_match_idx == y).float().mean()

        # logits for antmmf
        logits = torch.diag(similarity)
        logits = torch.reshape(logits, [logits.size(0), 1])
        pad = 1.0 - logits
        logits = torch.cat([pad, logits], dim=1)

        output = {
            "losses": {
                "contrastive_loss": loss,
            },
            "metrics": {
                "img_acc": img_acc,
                "txt_acc": txt_acc,
            },
            "logits": logits,
        }

        return output
