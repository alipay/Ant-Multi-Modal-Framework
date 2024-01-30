# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torch.nn.init import normal_, constant_

from antmmf.common import constants
from antmmf.datasets.features.vision.base_extractor import OnlineFeatureExtractor
from antmmf.models.s3dg import S3D
from antmmf.modules.vision.non_local import make_non_local
from antmmf.modules.vision.temporal_shift import make_temporal_shift

__all__ = ["VideoFeatureExtractor"]


class VideoFeatureExtractor(OnlineFeatureExtractor):
    MODEL_ARCH = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
    ]

    def __init__(
        self,
        model_arch,
        num_segments,
        modality,
        shift_div=4,
        shift_place="blockres",
        temporal_pool=False,
        add_non_local=False,
        pretrain=None,
        num_class=None,
        drop_out_rate=0.0,
        finetune=False,
        *args,
        **kwargs
    ):
        assert model_arch in VideoFeatureExtractor.MODEL_ARCH
        assert pretrain is None or pretrain in ["imagenet", "kinects"]
        assert modality in ["RGB", "RGBDiff", "Flow"]
        assert shift_place in ["block", "blockres"]
        self._model_arch = model_arch
        self.num_segments = num_segments
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.use_temporal_pool = temporal_pool
        self.add_non_local = add_non_local
        self.new_length = 1 if modality == "RGB" else 5
        self.num_class = num_class
        self.drop_out_rate = drop_out_rate
        self.modality = modality
        self.finetune = finetune
        self.pretrain = pretrain

        super().__init__(*args, **kwargs)

    def get_model_name(self):
        return "TSM_%s" % self._model_arch

    def get_feature_name(self):
        return "TSM_video_feature"

    def _build_preprocessor(self):
        def warpper(video_sample):
            if not isinstance(video_sample, torch.Tensor):
                assert isinstance(video_sample, np.ndarray)
                video_sample = torch.from_numpy(video_sample).double()
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
            if self.modality == "RGBDiff":
                sample_len = 3 * self.new_length
                video_sample = self._get_diff(video_sample)
            return video_sample.view((-1, sample_len) + video_sample.size()[-2:])

        return warpper

    def extract_features(self, video_sample):
        """
        Args:
            video_sample(np.ndArray): b,c,h,w, segment-based sampled video concatenated along channel-dim
        Returns:

        """
        sample = self._preprocessor(video_sample)
        with torch.no_grad():  # currently feature extractor params should not be involved in training
            features = self._extractor(sample)
            features = self._postprocessor(features)
        return features

    def _build_postprocessor(self):
        return lambda x: x

    def _build_extractor(self):
        if self.finetune and self.num_class is not None:
            base_model, new_fc = self._prepare_base_model_for_finetune(
                self.num_class, dropout=self.drop_out_rate
            )
        else:
            base_model, new_fc = self._prepare_base_model()
        if self.modality == "Flow":
            print("Converting the ImageNet model to a flow init model")
            base_model = self._construct_flow_model(base_model)
        elif self.modality == "RGBDiff":
            print("Converting the ImageNet model to RGB+Diff init model")
            base_model = self._construct_diff_model(base_model)
        return torch.nn.Sequential(base_model, new_fc)

    def _prepare_base_model(self):
        model = getattr(models, self._model_arch)(
            True if self.pretrain == "imagenet" else False
        )
        print("Adding temporal shift...")
        make_temporal_shift(
            model,
            self.num_segments,
            n_div=self.shift_div,
            place=self.shift_place,
            temporal_pool=self.use_temporal_pool,
        )
        if self.add_non_local:
            print("Adding non-local module...")
            make_non_local(model, self.num_segments)

        model.last_layer_name = "fc"
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Identity()
        return model, nn.Identity()

    def _prepare_base_model_for_finetune(self, num_class, dropout=0.0):
        model = self._prepare_base_model()
        fc_input_feature_dim = model.fc.in_features

        def _init(m, std=0.001):
            if isinstance(m, nn.Linear):
                normal_(m.weight, 0, std)
                constant_(m.bias, 0)

        if dropout > 0:
            model.fc = nn.Dropout(p=dropout)
            new_fc = nn.Linear(fc_input_feature_dim, num_class)
            _init(new_fc)

        else:
            model.fc = nn.Linear(fc_input_feature_dim, num_class)
            _init(model.fc)
            new_fc = nn.Identity()
        return model, new_fc

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(base_model.modules())
        first_conv_idx = list(
            filter(
                lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))
            )
        )[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = (
            params[0]
            .data.mean(dim=1, keepdim=True)
            .expand(new_kernel_size)
            .contiguous()
        )

        new_conv = nn.Conv2d(
            2 * self.new_length,
            conv_layer.out_channels,
            conv_layer.kernel_size,
            conv_layer.stride,
            conv_layer.padding,
            bias=True if len(params) == 2 else False,
        )
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][
            :-7
        ]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(base_model.modules())
        first_conv_idx = filter(
            lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))
        )[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = (
                params[0]
                .data.mean(dim=1, keepdim=True)
                .expand(new_kernel_size)
                .contiguous()
            )
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat(
                (
                    params[0].data,
                    params[0]
                    .data.mean(dim=1, keepdim=True)
                    .expand(new_kernel_size)
                    .contiguous(),
                ),
                1,
            )
            new_kernel_size = (
                kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]
            )

        new_conv = nn.Conv2d(
            new_kernel_size[1],
            conv_layer.out_channels,
            conv_layer.kernel_size,
            conv_layer.stride,
            conv_layer.padding,
            bias=True if len(params) == 2 else False,
        )
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][
            :-7
        ]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view(
            (
                -1,
                self.num_segments,
                self.new_length + 1,
                input_c,
            )
            + input.size()[2:]
        )
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = (
                    input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
                )
            else:
                new_data[:, :, x - 1, :, :, :] = (
                    input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
                )

        return new_data


class S3DGFeatureExtractor(OnlineFeatureExtractor):
    def __init__(self, s3d_model_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from antmmf.datasets.processors.video_processors import FMpegProcessor

        self.video_processor = FMpegProcessor()
        if s3d_model_path is not None:
            # weights downloaded from: https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
            model_data = torch.load(s3d_model_path)
            self.init_weight(self._extractor, model_data)

    def init_weight(self, module, state_dict, prefix=""):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        module._load_from_state_dict(
            state_dict, prefix, {}, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                self.init_weight(child, state_dict, prefix + name + ".")
        if len(missing_keys) > 0:
            print(
                "Weights of {} not initialized from pretrained model: {}".format(
                    self._extractor.__class__.__name__,
                    "\n   " + "\n   ".join(missing_keys),
                )
            )
        if len(unexpected_keys) > 0:
            print(
                "Weights from pretrained model not used in {}: {}".format(
                    self._extractor.__class__.__name__,
                    "\n   " + "\n   ".join(unexpected_keys),
                )
            )
        if len(error_msgs) > 0:
            print(
                "Weights from pretrained model cause errors in {}: {}".format(
                    self._extractor.__class__.__name__,
                    "\n   " + "\n   ".join(error_msgs),
                )
            )

    def get_model_name(self):
        return "s3dg"

    def get_feature_name(self):
        return "mixed5c"

    def get_feature_dim(self):
        return 1024

    def extract_features(self, video_sample, time_batch_size=1):
        """
        Args:
            video_sample(np.ndArray): b,c,h,w, segment-based sampled video concatenated along channel-dim
        Returns:
        """
        if isinstance(video_sample, (tuple, list)):
            assert len(video_sample) == 1
            video_sample = video_sample[0]

        video = self._preprocessor(video_sample)
        if torch.cuda.is_available():
            video = video.cuda()
        with torch.no_grad():  # currently feature extractor params should not be involved in training
            n_chunk = len(video)
            features = torch.FloatTensor(n_chunk, self.get_feature_dim()).fill_(0)
            n_iter = int(math.ceil(n_chunk / float(time_batch_size)))
            for i in range(n_iter):
                min_ind = i * time_batch_size
                max_ind = (i + 1) * time_batch_size
                video_batch = video[min_ind:max_ind]
                batch_features = self._extractor.forward_video(
                    video_batch, mixed5c=True
                )
                batch_features = F.normalize(batch_features, dim=1)
                features[min_ind:max_ind] = batch_features
            features = features.cpu().numpy()
            features = features.astype("float16")
            features = self._postprocessor(features)
        return features

    def _build_preprocessor(self):
        def warpper(video_path):  # video input: C x T x H x W
            video = self.video_processor({"image": video_path})[
                constants.VISION_MODALITY
            ]
            assert video.size(0) == 1, "batch_size must be 1 for s3dg"
            # Batch x 3 x T x H x W
            video = video.reshape(
                -1,
                3,
                self.video_processor.fps,
                self.video_processor.size,
                self.video_processor.size,
            )
            return video

        return warpper

    def _build_extractor(self):
        model = S3D(space_to_depth=True, with_text_module=False)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model

    def _build_postprocessor(self):
        return lambda x: x


if __name__ == "__main__":
    from antmmf.utils.general import get_absolute_path

    # extract tsm video feature
    video_sample = torch.randn(1, 24, 224, 224)
    video_feature_extrator = VideoFeatureExtractor(
        "resnet50", num_segments=8, modality="RGB"
    )
    feat = video_feature_extrator.extract_features(video_sample)
    assert list(feat.size()) == [video_feature_extrator.num_segments, 2048]

    # extract s3dg video feature
    s3dg_path = "~/Desktop/AntAI-MMF/work/VideoFeatureExtractor/model/s3d_howto100m.pth"
    sample_video = get_absolute_path("../tests/data/video/data/mp4/video9770.mp4")
    s3dg = S3DGFeatureExtractor(s3d_model_path=s3dg_path)
    ret = s3dg.extract_features(sample_video)
    assert list(ret.shape) == [11, 1024]
