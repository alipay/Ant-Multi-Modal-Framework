"""
# Copyright (c) 2023 Ant Group and its affiliates.

Contains the definition for Gated Separable 3D network (S3D-G).

Reference:
https://arxiv.org/pdf/1712.04851.pdf
"""
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from antmmf.common.constants import TEXT_MODALITY, VISION_MODALITY
from antmmf.common.registry import registry
from antmmf.models.base_model import BaseModel
from antmmf.utils.general import get_absolute_path


class InceptionBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        num_outputs_0_0a,
        num_outputs_1_0a,
        num_outputs_1_0b,
        num_outputs_2_0a,
        num_outputs_2_0b,
        num_outputs_3_0b,
        gating=True,
    ):
        super(InceptionBlock, self).__init__()
        self.conv_b0 = STConv3D(input_dim, num_outputs_0_0a, [1, 1, 1])
        self.conv_b1_a = STConv3D(input_dim, num_outputs_1_0a, [1, 1, 1])
        self.conv_b1_b = STConv3D(
            num_outputs_1_0a, num_outputs_1_0b, [3, 3, 3], padding=1, separable=True
        )
        self.conv_b2_a = STConv3D(input_dim, num_outputs_2_0a, [1, 1, 1])
        self.conv_b2_b = STConv3D(
            num_outputs_2_0a, num_outputs_2_0b, [3, 3, 3], padding=1, separable=True
        )
        self.maxpool_b3 = torch.nn.MaxPool3d((3, 3, 3), stride=1, padding=1)
        self.conv_b3_b = STConv3D(input_dim, num_outputs_3_0b, [1, 1, 1])
        self.gating = gating
        self.output_dim = (
            num_outputs_0_0a + num_outputs_1_0b + num_outputs_2_0b + num_outputs_3_0b
        )
        if gating:
            self.gating_b0 = SelfGating(num_outputs_0_0a)
            self.gating_b1 = SelfGating(num_outputs_1_0b)
            self.gating_b2 = SelfGating(num_outputs_2_0b)
            self.gating_b3 = SelfGating(num_outputs_3_0b)

    def forward(self, input):
        """Inception block"""
        b0 = self.conv_b0(input)
        b1 = self.conv_b1_a(input)
        b1 = self.conv_b1_b(b1)
        b2 = self.conv_b2_a(input)
        b2 = self.conv_b2_b(b2)
        b3 = self.maxpool_b3(input)
        b3 = self.conv_b3_b(b3)
        if self.gating:
            b0 = self.gating_b0(b0)
            b1 = self.gating_b1(b1)
            b2 = self.gating_b2(b2)
            b3 = self.gating_b3(b3)
        return torch.cat((b0, b1, b2, b3), dim=1)


class SelfGating(nn.Module):
    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        """Feature gating as used in S3D-G."""
        spatiotemporal_average = torch.mean(input_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = torch.sigmoid(weights)
        return weights[:, :, None, None, None] * input_tensor


class STConv3D(nn.Module):
    def __init__(
        self, input_dim, output_dim, kernel_size, stride=1, padding=0, separable=False
    ):
        super(STConv3D, self).__init__()
        self.separable = separable
        self.relu = nn.ReLU(inplace=True)
        assert len(kernel_size) == 3
        if separable and kernel_size[0] != 1:
            spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
            temporal_kernel_size = [kernel_size[0], 1, 1]
            if isinstance(stride, list) and len(stride) == 3:
                spatial_stride = [1, stride[1], stride[2]]
                temporal_stride = [stride[0], 1, 1]
            else:
                spatial_stride = [1, stride, stride]
                temporal_stride = [stride, 1, 1]
            if isinstance(padding, list) and len(padding) == 3:
                spatial_padding = [0, padding[1], padding[2]]
                temporal_padding = [padding[0], 0, 0]
            else:
                spatial_padding = [0, padding, padding]
                temporal_padding = [padding, 0, 0]
        if separable:
            self.conv1 = nn.Conv3d(
                input_dim,
                output_dim,
                kernel_size=spatial_kernel_size,
                stride=spatial_stride,
                padding=spatial_padding,
                bias=False,
            )
            self.bn1 = nn.BatchNorm3d(output_dim)
            self.conv2 = nn.Conv3d(
                output_dim,
                output_dim,
                kernel_size=temporal_kernel_size,
                stride=temporal_stride,
                padding=temporal_padding,
                bias=False,
            )
            self.bn2 = nn.BatchNorm3d(output_dim)
        else:
            self.conv1 = nn.Conv3d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.bn1 = nn.BatchNorm3d(output_dim)

    def forward(self, input):
        out = self.relu(self.bn1(self.conv1(input)))
        if self.separable:
            out = self.relu(self.bn2(self.conv2(out)))
        return out


def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding="SAME"):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == "SAME":
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Sentence_Embedding(nn.Module):
    def __init__(
        self,
        embd_dim,
        token_to_word_path,
        num_embeddings=66250,
        word_embedding_dim=300,
        word2vec_path="",
        max_words=16,
        output_dim=2048,
    ):
        super(Sentence_Embedding, self).__init__()
        has_pretrained_word2vec = False
        if word2vec_path is not None:
            if os.path.isfile(word2vec_path):
                self.word_embd = nn.Embedding.from_pretrained(torch.load(word2vec_path))
                has_pretrained_word2vec = True
        if not has_pretrained_word2vec:
            self.word_embd = nn.Embedding(num_embeddings, word_embedding_dim)
        self.fc1 = nn.Linear(word_embedding_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, embd_dim)
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = torch.zeros(size - len(tensor)).long()
            return torch.cat((tensor, zero), dim=0)

    def is_cuda(self):
        return self.fc1.bias.is_cuda

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [
            self.word_to_token[word] for word in words if word in self.word_to_token
        ]
        if words:
            we = self._zero_pad_tensor_token(torch.LongTensor(words), self.max_words)
            return we
        else:
            return torch.zeros(self.max_words).long()

    def words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text(sent)) for sent in x]
        return torch.stack(split_x, dim=0)

    def forward(self, x, raw_text=False):
        if raw_text:
            x = self.words_to_ids(x)
        with torch.no_grad():
            x = self.word_embd(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = torch.max(x, dim=1)[0]
        x = self.fc2(x)
        return x


class S3D(nn.Module):
    def __init__(
        self,
        num_classes=512,
        gating=True,
        space_to_depth=False,
        with_text_module=True,
        word2vec_path="",
        init="uniform",
        token_to_word_path="data/dict.npy",
        *config,
        **kwargs
    ):
        super(S3D, self).__init__()
        self.with_text_module = with_text_module
        self.num_classes = num_classes
        self.gating = gating
        self.space_to_depth = space_to_depth
        cr = kwargs.get("channel_reduce", 1)
        if space_to_depth:
            self.conv1 = STConv3D(
                24, 64 // cr, [2, 4, 4], stride=1, padding=(1, 2, 2), separable=False
            )
        else:
            self.conv1 = STConv3D(
                3, 64 // cr, [3, 7, 7], stride=2, padding=(1, 3, 3), separable=False
            )
        self.conv_2b = STConv3D(64 // cr, 64 // cr, [1, 1, 1], separable=False)
        self.conv_2c = STConv3D(
            64 // cr, 192 // cr, [3, 3, 3], padding=1, separable=True
        )
        self.gating = SelfGating(192 // cr)
        self.maxpool_2a = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="SAME"
        )
        self.maxpool_3a = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="SAME"
        )
        self.mixed_3b = InceptionBlock(
            *[c // cr for c in [192, 64, 96, 128, 16, 32, 32]]
        )
        self.mixed_3c = InceptionBlock(
            self.mixed_3b.output_dim, *[c // cr for c in [128, 128, 192, 32, 96, 64]]
        )
        self.maxpool_4a = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding="SAME"
        )
        self.mixed_4b = InceptionBlock(
            self.mixed_3c.output_dim, *[c // cr for c in [192, 96, 208, 16, 48, 64]]
        )
        self.mixed_4c = InceptionBlock(
            self.mixed_4b.output_dim, *[c // cr for c in [160, 112, 224, 24, 64, 64]]
        )
        self.mixed_4d = InceptionBlock(
            self.mixed_4c.output_dim, *[c // cr for c in [128, 128, 256, 24, 64, 64]]
        )
        self.mixed_4e = InceptionBlock(
            self.mixed_4d.output_dim, *[c // cr for c in [112, 144, 288, 32, 64, 64]]
        )
        self.mixed_4f = InceptionBlock(
            self.mixed_4e.output_dim, *[c // cr for c in [256, 160, 320, 32, 128, 128]]
        )
        self.maxpool_5a = self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding="SAME"
        )
        self.mixed_5b = InceptionBlock(
            self.mixed_4f.output_dim, *[c // cr for c in [256, 160, 320, 32, 128, 128]]
        )
        self.mixed_5c = InceptionBlock(
            self.mixed_5b.output_dim, *[c // cr for c in [384, 192, 384, 48, 128, 128]]
        )
        self.fc = nn.Linear(self.mixed_5c.output_dim, num_classes)
        if with_text_module:
            self.text_module = Sentence_Embedding(
                num_classes, token_to_word_path, word2vec_path=word2vec_path
            )

        if init == "kaiming_normal":
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _space_to_depth(self, inputs):
        B, C, T, H, W = inputs.shape
        inputs = inputs.view(B, C, T // 2, 2, H // 2, 2, W // 2, 2)
        inputs = inputs.permute(0, 3, 5, 7, 1, 2, 4, 6)
        inputs = inputs.contiguous().view(B, 8 * C, T // 2, H // 2, W // 2)
        return inputs

    def get_adv_parameters(self):
        # return parameters for adversarial genration and training
        # only do adversarial on the word-embeddings, not on others such as
        # positional embedding
        params = []
        for n, v in self.fc.named_parameters():
            params.append({"params": v, "name": n, "modality": VISION_MODALITY})
        for n, v in self.text_module.named_parameters():
            params.append({"params": v, "name": n, "modality": TEXT_MODALITY})

        return params

    def forward(self, video, text, mode="all", mixed5c=False):
        if mode == "all":
            assert self.with_text_module
            return self.forward_video(video), self.text_module(text)
        elif mode == VISION_MODALITY:
            return self.forward_video(video, mixed5c=mixed5c)
        elif mode == TEXT_MODALITY:
            assert self.with_text_module
            return self.text_module(text)
        else:
            raise NotImplementedError

    def forward_video(self, inputs, mixed5c=False):
        # out = {}
        if self.space_to_depth:
            inputs = self._space_to_depth(inputs)
        # 'Conv2d_1a_7x7'
        net = self.conv1(inputs)
        if self.space_to_depth:
            net = net[:, :, 1:, 1:, 1:]
        # out['Conv2d_1a_7x7'] = net
        # 'MaxPool_2a_3x3'
        net = self.maxpool_2a(net)
        # out['MaxPool_2a_3x3'] = net
        # 'Conv2d_2b_1x1'
        net = self.conv_2b(net)
        # out['Conv2d_2b_1x1'] = net
        # 'Conv2d_2c_3x3'
        net = self.conv_2c(net)
        # out['Conv2d_2c_3x3'] = net
        if self.gating:
            net = self.gating(net)
            # out['gating_1'] = net
        # 'MaxPool_3a_3x3'
        net = self.maxpool_3a(net)
        # out['MaxPool_3a_3x3'] = net
        # end_point = 'Mixed_3b'
        net = self.mixed_3b(net)
        # out['Mixed_3b'] = net
        # end_point = 'Mixed_3c'
        net = self.mixed_3c(net)
        # out['Mixed_3c'] = net
        # end_point = 'MaxPool_4a_3x3'
        net = self.maxpool_4a(net)
        # out['MaxPool_4a_3x3'] = net
        # end_point = 'Mixed_4b'
        net = self.mixed_4b(net)
        # out['Mixed_4b'] = net
        # end_point = 'Mixed_4c'
        net = self.mixed_4c(net)
        # out['Mixed_4c'] = net
        # end_point = 'Mixed_4d'
        net = self.mixed_4d(net)
        # out['Mixed_4d'] = net
        # end_point = 'Mixed_4e'
        net = self.mixed_4e(net)
        # out['Mixed_4e'] = net
        # end_point = 'Mixed_4f'
        net = self.mixed_4f(net)
        # out['Mixed_4f'] = net
        # end_point = 'MaxPool_5a_2x2'
        net = self.maxpool_5a(net)
        # out['MaxPool_5a_2x2'] = net
        # end_point = 'Mixed_5b'
        net = self.mixed_5b(net)
        # out['Mixed_5b'] = net
        # end_point = 'Mixed_5c'
        net = self.mixed_5c(net)
        # out['Mixed_5c'] = net
        # out['Avgpool'] = net
        net = torch.mean(net, dim=[2, 3, 4])
        if mixed5c:
            return net
        net = self.fc(net)
        # out['final'] = net
        return net


@registry.register_model("s3d")
class S3DModel(BaseModel):
    """
    TODO: support multiple modalities, currently assume text is one of the modality
    """

    def __init__(self, config):
        super().__init__(config)
        self._datasets = []
        for _, attr in registry.get("config").task_attributes.items():
            for dataset in attr.dataset_attributes:
                self._datasets.append(dataset)
        self._mixed_5c = config.get("mixed_5c", False)
        self._modalities = config.get("modalities")

    def build(self):
        pretrained_model_dir = self.config.get("pretrained_model_dir")
        pretrain_cnn_path = self.config.pretrain_cnn_path
        if pretrained_model_dir is not None:
            for k in ["token_to_word_path", "word2vec_path"]:
                self.config.params[k] = get_absolute_path(
                    os.path.join(pretrained_model_dir, self.config.params[k])
                )
            pretrain_cnn_path = get_absolute_path(
                os.path.join(pretrained_model_dir, pretrain_cnn_path)
            )

        self.model = S3D(**self.config.params)

        if os.path.isfile(pretrain_cnn_path):
            net_data = torch.load(pretrain_cnn_path)
            self.model.load_state_dict(net_data)

    def forward(self, sample_list):
        text = (
            sample_list.get(self._modalities[1]) if len(self._modalities) == 2 else None
        )
        video = sample_list.get(self._modalities[0])

        assert (
            text is not None or video is not None
        ), "must have one of modality has valid observation"

        mode = sample_list["mode"]
        if isinstance(mode, list):
            mode = mode[0]
        if text is None:
            mode = VISION_MODALITY
        if video is None:
            mode = TEXT_MODALITY

        if len(video.shape) == 5:
            video = video.unsqueeze(1)
            # add a dimension of num of clips if that is missing from inputs
        video = video.view(
            -1, video.shape[2], video.shape[3], video.shape[4], video.shape[5]
        )

        y = self.model(video, text, mode, mixed5c=self._mixed_5c)

        text = None
        video = None
        if mode == "all":
            video = y[0]
            text = y[1]
        if mode == TEXT_MODALITY:
            text = y
        if mode == VISION_MODALITY:
            video = y

        ret = {}
        if text is not None:
            ret.update({TEXT_MODALITY: text})
        if video is not None:
            ret.update({VISION_MODALITY: video})

        return ret
