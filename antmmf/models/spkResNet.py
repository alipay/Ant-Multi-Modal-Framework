# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from antmmf.common.registry import registry
from antmmf.structures import Sample, SampleList
from antmmf.models.base_model import BaseModel
from antmmf.common import Configuration
from antmmf.utils.general import get_antmmf_root
from antmmf.utils.logger import Logger


# basic components of ResNet
def conv1x1(in_planes, out_planes, stride=1, padding=0):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=False
    )


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.expansion = 1
        self.conv2 = conv3x3(planes, planes * self.expansion, stride=1)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        self.nonlinear = nn.ELU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, planes * self.expansion, stride),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.nonlinear(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = self.nonlinear(out)
        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.nonlinear = nn.ELU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, planes * self.expansion, stride),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.nonlinear(self.bn1(self.conv1(x)))
        out = self.nonlinear(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = self.nonlinear(out)
        return out


class ResNetWithNoConv1(nn.Module):
    """
    modified ResNet, removing traditional conv1 & max_pooling layer
    """

    def __init__(self, block, in_planes=64, num_blocks=[3, 4, 6, 3]):
        super(ResNetWithNoConv1, self).__init__()
        self.in_planes = in_planes

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride]
        if num_blocks > 1:
            strides += [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    @staticmethod
    def get_output_channels():
        return 512


class NormLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features), requires_grad=True
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        norm_x = F.normalize(x, p=2, dim=-1)
        norm_w = F.normalize(self.weight, p=2, dim=-1)
        y = F.linear(norm_x, norm_w)
        return y


@registry.register_model("SpkResNet")
class SpkResNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # num_classes default value from registry
        num_classes = registry.get(self.config.registry_key_of_num_classes, None)
        # enable override `num_classes` in config.yml
        num_classes = self.config.get("num_classes", num_classes)
        assert (
            num_classes is not None
        ), "num_classes is not indicated by neither config.num_classes nor registry"

        use_norm_linear = self.config.use_norm_linear
        resnet_type = self.config.resnet_type

        self.front_end_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=(1, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
        )

        if resnet_type == "resnet34":
            num_blocks = [3, 4, 6, 3]
            self.write("Using 34-layer resnet")
        elif resnet_type == "resnet18":
            num_blocks = [2, 2, 2, 2]
            self.write("Using 18-layer resnet")
        else:
            raise RuntimeError("Unknown resnet type {}".format(resnet_type))

        output_channels = ResNetWithNoConv1.get_output_channels()
        self.resnet_layer = nn.Sequential(
            ResNetWithNoConv1(block=BasicBlock, in_planes=64, num_blocks=num_blocks),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                stride=(1, 2),
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ELU(inplace=True),
        )

        self.backend = nn.Sequential(
            nn.Linear(output_channels * 2 * 2, self.config.projection_dim, bias=True),
            nn.BatchNorm1d(self.config.projection_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.config.projection_dim, self.config.embedding_dim, bias=True),
            nn.BatchNorm1d(self.config.embedding_dim),
            nn.ELU(inplace=True),
        )

        self.classifier = (
            NormLinear(self.config.embedding_dim, num_classes)
            if (use_norm_linear is True)
            else nn.Linear(self.config.embedding_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build(self):
        return None

    def write(self, info):
        if self.writer:
            self.writer.write(info)
        else:
            print(info)

    def get_custom_scheduler(self, trainer):
        base_lr = trainer.config.optimizer_attributes.params.lr
        step_size_up = (
            trainer.epoch_iterations * 2 if "train" in trainer.run_type else 2000
        )
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            trainer.optimizer,
            base_lr=base_lr * 0.25,
            max_lr=base_lr * 1.0,
            step_size_up=step_size_up,
            cycle_momentum=False,
        )
        return lr_scheduler

    def forward2embed(self, x):
        # batch_size, 1, num_frames, frame_feature_dim(80)
        out = x.unsqueeze(1)
        out = self.front_end_layer(out)  # batch_size, 64, 18, 39
        out = self.resnet_layer(out)  # batch_size, 512, 1, 2
        ###
        out = out.transpose(1, 2).contiguous()  # batch_size, 1, 512, 2
        out = out.view(out.shape[0], out.shape[1], -1)  # batch_size, 1, 1024
        # statistics pooling
        mean = torch.mean(out, 1)
        std = torch.std(out, 1)
        out = torch.cat((mean, std), 1)  # batch_size, 2048
        ###
        out = self.backend(out)  # batch_size, 512
        return out

    def forward(self, sample_list):
        """

        Args:
            sample_list: batch_size, num_frames, frame_feature_dim(80)

        Returns:

        """
        input = sample_list.image
        embed = self.forward2embed(input)  # batch_size, 512
        logits = self.classifier(embed)  # batch_size, num_classes
        res = {"out_feat": embed, "logits": logits}
        res["prob"] = F.softmax(res["logits"], dim=-1)
        return res

    def embed_dim_size(self):
        return self.config.embedding_dim
