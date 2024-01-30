# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import types
import warnings
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.checkpoint import checkpoint

from antmmf.common import Configuration, configurable
from antmmf.modules.embeddings import (
    DetrPositionEmbeddingSine,
    DetrPositionEmbeddingLearned,
)
from antmmf.modules.layers import FrozenBatchNorm2d
from antmmf.modules.module_registry import ModuleRegistry
from antmmf.modules.vision.backbone.clip.model import CLIPImageEncoder
from antmmf.modules.vision.backbone.efficientnet import BatchEfficientNetImageEncoder
from antmmf.modules.vision.backbone.pvt import BatchPVTEncoder
from antmmf.modules.vision.backbone.pvt import PVTImageEncoder
from antmmf.modules.vision.backbone.video_swin import SwinTransformer3D
from antmmf.modules.vision.non_local import make_non_local
from antmmf.modules.vision.temporal_shift import make_temporal_shift
from antmmf.structures import NestedTensor
from antmmf.utils.general import get_transformer_model_vocab_path
from .image_feature_encoder import FinetuneFasterRcnnFpnFc7


class VisualEncoder(ModuleRegistry):
    """
    A graph encoder register for visual encoder, all other details can be
    seen from :class:`antmmf.modules.module_registry.ModuleRegistry`.

    Args:
        config (Configuration): configuration of visual encoder.
    """

    def __init__(self, config: Configuration, *args, **kwargs):
        # compatible codes, and they will be removed in the future.

        self.encoder_type = config.type
        config_kwargs = config.get("params", {})
        super(VisualEncoder, self).__init__(
            config.type, *args, **kwargs, **config_kwargs
        )

    def check_input(self, input_img):
        # check whether input_img for `forward` func is valid
        if self.encoder_type in [
            "Identity",
            "resnet152",
            "FinetuneFasterRcnnFpnFc7",
        ]:
            assert len(input_img.size()) == 4, "input_image size = [Bx3xHxW]"
            assert input_img.size()[1] == 3, "input_image size = [Bx3xHxW]"
        elif self.encoder_type == "BatchImageEncoder":
            assert len(input_img.size()) == 5, "input_image size = [Bxnum_imagesx3xHxW]"
            # assert input_img.size()[2] == 3, "input_image size = [Bxnum_imagesx3xHxW]"
        elif self.encoder_type in ["VideoTSMEncoder", "ImageVideoEncoder"]:
            assert (
                len(input_img.size()) == 4
            ), "input_image size = [Bx3*num_segmentsxHxW]"
            assert (
                input_img.size()[1] % 3 == 0
            ), "input_image size = [Bx3*num_segmentsxHxW]"
        else:
            warnings.warn(
                f"Unable to check input shape for ImageEncoder:{self.encoder_type}"
            )


VisualEncoder.register(FinetuneFasterRcnnFpnFc7)
VisualEncoder.register(BatchEfficientNetImageEncoder)
VisualEncoder.register(BatchPVTEncoder)
VisualEncoder.register(CLIPImageEncoder)


@VisualEncoder.register()
class ResNetImageEncoder(nn.Module):
    """
    Encode a single image with torchvision ResNet series:
        type: resnet18/resnet50/resnet101/resnet152
          params:
            pretrained: false
            pool_type: avg
            num_output_features: 8
            freeze: False # if image encoder are trainable
            freeze_bn: False # whether freeze BN during training
            replace_stride_with_dilation: [False, False, False]
            gradient_checkpointing: false # use gradient checkpoint to save memory
    """

    @configurable
    def __init__(
        self,
        encoder_type: str,
        pool_type: str,
        num_output_features: int,
        gradient_checkpointing: List[bool] = None,
        pretrained: bool = True,
        freeze: bool = False,
        freeze_bn: bool = False,
        replace_stride_with_dilation: List[bool] = None,
        input_size: Optional[int] = None,  # needed for pooling
        **kwargs,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.pretrained = pretrained
        self.gradient_checkpointing = gradient_checkpointing
        self.freeze_bn = freeze_bn
        self.replace_stride_with_dilation = replace_stride_with_dilation

        self.out_dim = 512 if encoder_type in ["resnet18", "resnet34"] else 2048
        self.model = self.get_model()

        # change adaptive pooling to normal pooling, since ONNX does not support this op
        self.input_size = input_size
        self.pool_type = pool_type
        # check if trainable
        if freeze:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(False)

        # -1 will keep the original feature size
        if num_output_features == -1:
            outsz = -1
        elif num_output_features in [1, 2, 3, 5, 7]:
            outsz = (num_output_features, 1)
        elif num_output_features == 4:
            outsz = (2, 2)
        elif num_output_features == 6:
            outsz = (3, 2)
        elif num_output_features == 8:
            outsz = (4, 2)
        elif num_output_features == 9:
            outsz = (3, 3)
        else:
            raise Exception("num_output_features only has 1 to 9")

        self.pool = self.get_pool_module(outsz)

    def get_pool_module(self, outputsz):
        pool_module = nn.Identity()
        if outputsz != -1:
            if self.input_size is not None:
                isz = self.input_size // 32
                outputsz = np.array(outputsz)
                inputsz = np.array([isz, isz])
                stridesz = np.floor(inputsz / outputsz).astype(np.int32)
                kernelsz = inputsz - (outputsz - 1) * stridesz
                pool_func = nn.AvgPool2d if self.pool_type == "avg" else nn.MaxPool2d
                pool_module = pool_func(
                    kernel_size=list(kernelsz), stride=list(stridesz)
                )
            else:
                pool_func = (
                    nn.AdaptiveAvgPool2d
                    if self.pool_type == "avg"
                    else nn.AdaptiveMaxPool2d
                )
                pool_module = pool_func(outputsz)
        return pool_module

    def get_model(self):
        model = self.build_model()
        if self.gradient_checkpointing is not None:
            del model.fc, model.avgpool
            model = self.get_resnet_support_gradient_ckpt(model)
        else:
            modules = list(model.children())[:-2]
            model = nn.Sequential(*modules)
        return model

    def get_resnet_support_gradient_ckpt(self, model):
        # use gradient checkpoint to save memory

        def _forward_impl(self, x, gcs=self.gradient_checkpointing):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            # original resnet implementation
            # x = self.layer1(x)
            # x = self.layer2(x)
            # x = self.layer3(x)
            # x = self.layer4(x)
            # x = self.avgpool(x)
            # x = torch.flatten(x, 1)
            # x = self.fc(x)
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for gc, layer in zip(gcs, layers):
                if gc is True:
                    x = checkpoint(layer, x)
                else:
                    x = layer(x)
            return x

        model._forward_impl = types.MethodType(_forward_impl, model)
        return model

    def __setstate__(self, state):
        """
        For maya inference, unpickling ResNetImageEncoder will call torchvision's `forward` method
        by default, thus will raise AttributeError: 'ResNet' object has no attribute 'avgpool'.
        Reloading self.model to override the default `forward` method will avoid this problem.
        """
        self.__dict__ = state
        if self.gradient_checkpointing is not None:
            self.model = self.get_resnet_support_gradient_ckpt(self.model)

    def build_model(self):
        # freeze BatchNormalization during training due to small image batch_size in
        # multi-modal scenario FrozenBatchNorm2d
        norm_layer = FrozenBatchNorm2d if self.freeze_bn else None

        # use dilation conv for higher resolution
        # eg. replace_stride_with_dilation = [False, False, True] to replace
        # C5 layer into dialation conv, so the output will be downsampled to 1/16.
        replace_stride_with_dilation = self.replace_stride_with_dilation

        return getattr(torchvision.models, self.encoder_type)(
            pretrained=self.pretrained,
            norm_layer=norm_layer,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        if x.ndim == 5:  # B,N,3,224,224
            x = x.squeeze(1)
        y = self.model(x)
        out = self.pool(y)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048


@VisualEncoder.register()
class DetrImageEncoder(ResNetImageEncoder):
    """
    Encode batch images with torchvision ResNet series:

    .. code-block:: python

        type: DetrImageEncoder
          params:
            encoder_type: resnet18/resnet50/resnet101/resnet152
            pretrained: true
            pool_type: avg
            num_output_features: -1
            freeze: False # if image encoder are trainable
            freeze_bn: True # whether freeze BN during training
            replace_stride_with_dilation: [False, False, True]
            output_channels: null
            with_position_embedding: True

    """

    @configurable
    def __init__(
        self,
        position_embedding: Configuration,
        with_position_embedding: bool,
        output_channels: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.with_position_embedding = with_position_embedding

        # add detr positional Embedding
        if self.with_position_embedding:
            position_embedding_type = position_embedding.type
            position_embedding_params = position_embedding.params
            assert position_embedding_type in ["sine", "learned"]

            self.position_embedding = (
                DetrPositionEmbeddingSine(**position_embedding_params)
                if position_embedding_type == "sine"
                else DetrPositionEmbeddingLearned(**position_embedding_params)
            )

        self._add_channel_projection = (
            output_channels is not None and output_channels != self.out_dim
        )
        if self._add_channel_projection:
            self.output_proj = nn.Conv2d(self.out_dim, output_channels, kernel_size=1)
            self.out_dim = output_channels

    def forward(self, image, image_mask):
        """
        :param image(torch.float32): [b, N, channels, h, w]
        :param image_mask(torch.bool): [b, N, h, w], with True indicating padding areas.
        :return:
            out_feats(torch.float32): [b, N, self.out_dim, h//32, w//32]
            out_pos(torch.float32): [b, N, 2*position_embedding.params.num_pos_feats, h//32, w//32]
            out_feat_masks(torch.bool): [b, N, h//32, w//32]
        """
        bsz, num_imgs = image.size(0), image.size(1)
        # Detr images are large, do batch inference may cause OOM of gpu memory.
        out_feats, out_pos, out_feat_masks = [], [], []
        for idx in range(num_imgs):
            img_feat = self.model(image[:, idx, ...])
            img_feat_mask = F.interpolate(
                image_mask[:, idx, ...][None].float(), size=img_feat.shape[-2:]
            ).to(torch.bool)[0]
            if self.with_position_embedding:
                group_feat_mask = NestedTensor(img_feat, img_feat_mask)
                img_pos = self.position_embedding(group_feat_mask).to(img_feat.dtype)
                out_pos += [img_pos]
            if self._add_channel_projection:
                img_feat = self.output_proj(img_feat)
            out_feats += [img_feat]
            out_feat_masks += [img_feat_mask]

        output_dict = dict(
            grid_feature=torch.stack(out_feats).transpose(0, 1),
            grid_mask=torch.stack(out_feat_masks).transpose(0, 1),
            grid_pos=torch.stack(out_pos).transpose(0, 1)
            if self.with_position_embedding
            else None,
        )
        return output_dict


@VisualEncoder.register()
class VideoSwinEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        pretrained: bool,
        freeze: bool = False,
        gradient_checkpointing: bool = None,
        model_path: str = None,
        **kwargs,
    ):
        super().__init__()
        self.out_dim = 768

        self.swin = SwinTransformer3D.from_pretrained(
            weights_path=model_path,
            pretrained=pretrained,
            use_checkpoint=gradient_checkpointing,
        )

        # check if trainable
        if freeze:
            for name, parameter in self.swin.named_parameters():
                parameter.requires_grad_(False)

        # self.emb_cls = nn.Parameter(0.02 * torch.randn(1, 1, 1, 768))
        # self.emb_pos = nn.Parameter(
        #     0.02 * torch.randn(1, 1, 1 + 14 ** 2, 768)
        # )  # 输入图片最大分辨率 448*448
        # self.emb_len = nn.Parameter(0.02 * torch.randn(1, 6, 1, 768))  # 最多6帧
        # self.norm = nn.LayerNorm(768)

    def forward(self, image, image_mask):
        """
        :param image(torch.float32): [b, N, channels, h, w]
        :param image_mask(torch.bool): [b, N, h, w], with True indicating padding areas.
        :return:
            out_feats(torch.float32): [b, N, self.out_dim, h//32, w//32]
            out_pos(torch.float32): [b, N, 2*position_embedding.params.num_pos_feats, h//32, w//32]
            out_feat_masks(torch.bool): [b, N, h//32, w//32]
        """
        _B, _T, _C, _H, _W = image.shape
        _h, _w = _H // 32, _W // 32

        img_feat = self.swin(image.transpose(1, 2)).transpose(
            1, 2
        )  # [_B, _T, 768, _h,  _w]
        img_mask = F.interpolate(image_mask.float(), size=(_h, _w)).to(torch.bool)

        # f_img = img_feat.permute(0, 1, 3, 4, 2).view([_B, _T, _h * _w, 768])
        # # add [cls] token
        # f_img = torch.cat([self.emb_cls.expand([_B, _T, -1, -1]), f_img], dim=2)
        # # add frame position embedding and temporal embedding
        # f_img += (
        #     self.emb_pos.expand([_B, _T, -1, -1])[:, :, : 1 + _h * _w, :]
        #     + self.emb_len.expand([_B, -1, 1 + _h * _w, -1])[:, :_T, :, :]
        # )
        # # normalization
        # f_img = self.norm(f_img).view([_B, _T * (1 + _h * _w), -1])

        output_dict = dict(
            grid_feature=img_feat,
            grid_mask=img_mask,
            grid_feature_with_pos=None,
        )
        return output_dict


@VisualEncoder.register()
class DetrBatchPVTImageEncoder(PVTImageEncoder):
    """
    Encode batch image with pvt
    """

    @configurable
    def __init__(self, output_channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_proj = nn.Conv2d(self.out_dim, output_channels, kernel_size=1)

    def forward(self, image, image_mask):
        """
        :param image(torch.float32): [b, N, channels, h, w]
        :param image_mask(torch.bool): [b, N, h, w], with True indicating padding areas.
        :return:
            out_feats(torch.float32): [b, N, self.out_dim, h//32, w//32]
            out_pos(torch.float32): [b, N, 2*position_embedding.params.num_pos_feats, h//32, w//32]
            out_feat_masks(torch.bool): [b, N, h//32, w//32]
        """
        x = image
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        bsz, num_imgs = x.size(0), x.size(1)  # B*Nx3x224x224

        out_feats, out_feat_masks = [], []
        for idx in range(num_imgs):
            _, img_feat = self.model.forward_features(
                image[:, idx, ...], gcs=self.gradient_checkpointing
            )
            out_feat = self.output_proj(img_feat)
            img_feat_mask = F.interpolate(
                image_mask[:, idx, ...][None].float(), size=out_feat.shape[-2:]
            ).to(torch.bool)[0]
            out_feats += [out_feat]
            out_feat_masks += [img_feat_mask]
        out_feats = torch.stack(out_feats).transpose(0, 1)
        out_feat_masks = torch.stack(out_feat_masks).transpose(0, 1)

        output_dict = dict(
            grid_feature=out_feats, grid_mask=out_feat_masks, grid_pos=None
        )
        return output_dict


@VisualEncoder.register()
class VideoTSMEncoder(ResNetImageEncoder):
    """
    encode video segments using TSM method
    config example:

    .. code-block:: python

        type: VideoTSMEncoder
        params:
            base_model: resnet50
            pretrain_from: null # kinetics or  imagenet or null
            num_segments: 8 # sampled video segments
            shift_div: 4 # 1/4 of channels are shifted
            shift_place: 'blockres' # 'blockres' is always better than 'block'
            temporal_pool: false
            non_local: false # add non-local block
            pool_type: 'avg'
            num_output_features: 8

    """

    @configurable
    def __init__(
        self,
        base_model: str,
        pretrain_from: str,
        num_segments: int,
        shift_div: int,
        shift_place: int,
        temporal_pool: bool,
        non_local: bool,
        *args,
        **kwargs,
    ):
        self.base_model = base_model
        self.num_segments = num_segments
        self.pretrain_from = pretrain_from
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        super().__init__(*args, **kwargs)

    def forward(self, video_segments):
        # Bxnum_segments*3x244x244 -> B*num_segmentsx3x224x224
        video_segments = video_segments.view((-1, 3) + video_segments.size()[-2:])
        x = self.model(video_segments)  # B*num_segmentsx2048x7x7
        x = x.view((-1, self.num_segments) + x.size()[1:])  # Bxnum_segmentsx2048x7x7
        x = x.mean(dim=1, keepdim=False)  # Bx2048x7x7
        out = self.pool(x)
        out = torch.flatten(out, start_dim=2)  # Bx2048xnum_output_features
        out = out.transpose(1, 2).contiguous()  # Bxnum_output_featuresx2048
        return out

    def build_model(self):
        model = getattr(torchvision.models, self.base_model)(
            True if self.pretrain_from == "imagenet" else False
        )
        make_temporal_shift(
            model,
            self.num_segments,
            n_div=self.shift_div,
            place=self.shift_place,
            temporal_pool=self.temporal_pool,
        )
        if self.non_local:
            make_non_local(model, self.num_segments)
        if self.pretrain_from and self.pretrain_from != "imagenet":
            pretrain_path = get_transformer_model_vocab_path(self.pretrain_from)
            ckpt = torch.load(pretrain_path, map_location=torch.device("cpu"))
            sd = ckpt["state_dict"]
            load_dict = {}
            for k, v in sd.items():
                load_dict[k.replace("module.base_model.", "")] = v
            model.load_state_dict(load_dict, strict=False)
        return model


@VisualEncoder.register()
class ImageVideoEncoder(nn.Module):
    """
    An enhanced VideoTSMEncoder: temporal_feat + image_feat

    Config.pretrain_from can set to 'imagenet'(torchvision weights) or
    'TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth'
    which can be downloaded from:
    https://file.lzhu.me/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
    """

    @configurable
    def __init__(
        self,
        encoder_type: str = "ImageVideoEncoder",
        base_model: str = "resnet50",
        pretrain_from: str = "imagenet",  #
        num_segments: int = 8,
        shift_div: int = 4,
        shift_place: str = "blockres",
        temporal_pool: bool = False,
        non_local: bool = False,
        pool_type: str = "avg",
        num_output_features: int = 8,  # equal to num_segments
    ):
        super().__init__()
        self.config = Configuration(locals())

        self.video_encoder = self.build_video_encoder()
        self.image_encoder = self.build_image_encoder()
        pool_func = nn.AdaptiveAvgPool2d if pool_type == "avg" else nn.AdaptiveMaxPool2d
        self.pool = pool_func((1, 1))

    def forward(self, video_segments):
        # Bxnum_segments*3x244x244 -> B*num_segmentsx3x224x224
        batch_size, num_segments = video_segments.shape[0], video_segments.shape[1] // 3
        video_segments = video_segments.view((-1, 3) + video_segments.size()[-2:])
        temporal_feat = self.pool(
            self.video_encoder(video_segments)
        )  # B*num_segmentsx2048x7x7
        image_feat = self.pool(
            self.image_encoder(video_segments)
        )  # B*num_segmentsx2048x7x7

        assert image_feat.shape[-3] == temporal_feat.shape[-3]

        feat_dim = image_feat.shape[-3]
        feat = (image_feat + temporal_feat).reshape(
            [batch_size, feat_dim, num_segments, -1]
        )
        # Bx2048xnum_segmentsx7x7
        out = torch.flatten(feat, start_dim=2)  # Bx2048xnum_output_features
        out = (
            out.transpose(1, 2).contiguous().unsqueeze(1)
        )  # Bxnum_segmentsxnum_output_featuresx2048
        return out

    @property
    def out_dim(self):
        return 2048

    def build_image_encoder(self):
        model = getattr(torchvision.models, self.config.base_model)(True)
        return nn.Sequential(*list(model.children())[:-2])

    def build_video_encoder(self):
        model = getattr(torchvision.models, self.config.base_model)(
            True if self.config.pretrain_from == "imagenet" else False
        )
        make_temporal_shift(
            model,
            self.config.num_segments,
            n_div=self.config.shift_div,
            place=self.config.shift_place,
            temporal_pool=self.config.temporal_pool,
        )
        if self.config.non_local:
            make_non_local(model, self.config.num_segments)
        if self.config.pretrain_from and self.config.pretrain_from != "imagenet":
            pretrain_path = get_transformer_model_vocab_path(self.config.pretrain_from)
            ckpt = torch.load(pretrain_path, map_location=torch.device("cpu"))
            sd = ckpt["state_dict"]
            load_dict = {}
            for k, v in sd.items():
                load_dict[k.replace("module.base_model.", "")] = v
            model.load_state_dict(load_dict, strict=False)
        return nn.Sequential(*list(model.children())[:-2])


@VisualEncoder.register()
class BatchImageEncoder(ResNetImageEncoder):
    """
    Encode batch images with torchvision ResNet series:
    eg.

    .. code-block:: python

        type: BatchImageEncoder
          params:
            encoder_type: resnet18/resnet50/resnet101/resnet152
            pretrained: false
            pool_type: avg
            num_output_features: 8

    """

    def forward(self, x):
        # BxNx3x224x224 -> BxNxconfig.num_output_featuresx2048
        bsz, num_imgs = x.size(0), x.size(1)
        x = x.view([-1] + list(x.shape)[2:])  # B*Nx3x224x224
        out = super().forward(x)  # B*Nxnum_featuresx2048
        return out.view([bsz, num_imgs] + list(out.shape[1:])).contiguous()
