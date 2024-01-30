# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from antmmf.datasets.features.vision.base_extractor import OnlineFeatureExtractor

__all__ = [
    "ClassificationFeatureExtractor",
    "ResNetFeatureExtractor",
    "DenseNetFeatureExtractor",
    "InceptionV3FeatureExtractor",
]


class ClassificationFeatureExtractor(OnlineFeatureExtractor):
    """
    Class wrapper for classification models in torchvision, see detail at:
    https://pytorch.org/docs/stable/torchvision/models.html
    """

    TARGET_IMAGE_SIZE = [448, 448]
    CHANNEL_MEAN = [0.485, 0.456, 0.406]
    CHANNEL_STD = [0.229, 0.224, 0.225]

    def __init__(
        self, required_feature_shape=None, use_pratrained=True, *args, **kwargs
    ):
        self._feat_shape = required_feature_shape
        self._pratrained = use_pratrained

        super().__init__(*args, **kwargs)

    def extract_features(self, pil_imgs):
        """
        Args:
            pil_img(list): list of RGB mode PIL.Image object, needed to be channels_first

        Returns:

        """
        if not isinstance(pil_imgs, (list, tuple)):  # support batch inference
            pil_imgs = [pil_imgs]
        img_tensor = []
        for pil_img in pil_imgs:
            img_transform = self._preprocessor(pil_img)
            if img_transform.shape[0] == 1:
                img_transform = img_transform.expand(3, -1, -1)
            img_tensor.append(img_transform)
        img_transforms = torch.stack(img_tensor, 0)
        if torch.cuda.is_available():
            img_transforms = img_transforms.to("cuda")
        with torch.no_grad():
            features = self._extractor(img_transforms)
            features = self._postprocessor(features)
        return features

    def _build_extractor(self):
        raise NotImplementedError(
            "ClassificationFeatureExtractor doesn't implement a _build_extractor method"
        )

    def _build_preprocessor(self):
        return transforms.Compose(
            [
                transforms.Resize(self.__class__.TARGET_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    self.__class__.CHANNEL_MEAN, self.__class__.CHANNEL_STD
                ),
            ]
        )

    def _build_postprocessor(self):
        def wrapper(x):
            x = x.permute(0, 2, 3, 1)  # b, c, h, w -> b, h, w, c
            assert x.numel() == np.prod(
                self._feat_shape
            ), "feature shape:{} does not match required_feature_shape:{}".format(
                list(x.shape), self._feat_shape
            )
            return x.view(*self._feat_shape)

        return wrapper if self._feat_shape is not None else lambda x: x

    def print_extractor_summary(self):
        from torchsummary import summary

        summary(self._extractor, tuple([3] + self.__class__.TARGET_IMAGE_SIZE))


class ResNetFeatureExtractor(ClassificationFeatureExtractor):
    """
    for resnet18 and resnet34 model, default extracted feature_size will be:
    [1, 14, 14, 512]
    for other resnet models, extracted feature_size will be:
    [1, 14, 14, 2048]

    Usage::

        >>> expected_feat_size = [1,14,14,2048]
        >>> extractor = ResNetFeatureExtractor('resnet152')
        >>> features = extractor.extract_features(pil_image)
        >>> assert list(features.shape)==expected_feat_size
    """

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

    def __init__(self, model_arch, *args, **kwargs):
        assert model_arch in ResNetFeatureExtractor.MODEL_ARCH
        self._model_arch = model_arch
        super().__init__(*args, **kwargs)

    def get_model_name(self):
        return self._model_arch

    def get_feature_name(self):
        return "block5_feature"

    def _build_extractor(self):
        resnet = getattr(models, self._model_arch)(pretrained=self._pratrained)
        resnet.eval()
        modules = list(resnet.children())[:-2]
        resnet_model = torch.nn.Sequential(*modules)
        if torch.cuda.is_available():
            resnet_model.to("cuda")
        return resnet_model


class DenseNetFeatureExtractor(ClassificationFeatureExtractor):
    """
    https://arxiv.org/abs/1608.06993
    for DenseNet, feature_size may vary due to various model architecture.
    model and its expected feature_size are as following:
    densenet121 -> [1,14,14,1024]
    densenet169 -> [1,14,14,1664]
    densenet201 -> [1,14,14,1920]
    densenet161 -> [1,14,14,2208]
    Usage::
        >>> expected_feat_size = [1,14,14,1024]
        >>> extractor = DenseNetFeatureExtractor('densenet121')
        >>> features = extractor.extract_features(pil_image)
        >>> assert list(features.shape)==expected_feat_size
    """

    MODEL_ARCH = ["densenet121", "densenet169", "densenet201", "densenet161"]

    def __init__(self, model_arch, *args, **kwargs):
        assert model_arch in DenseNetFeatureExtractor.MODEL_ARCH
        self._model_arch = model_arch
        super().__init__(*args, **kwargs)

    def get_model_name(self):
        return self._model_arch

    def get_feature_name(self):
        return "DenseBlock4"

    def _build_extractor(self):
        net = getattr(models, self._model_arch)(pretrained=self._pratrained)
        net.eval()
        model = net.features
        if torch.cuda.is_available():
            model.to("cuda")
        return model


class InceptionV3FeatureExtractor(ClassificationFeatureExtractor):
    """
    In contrast to the other models the inception_v3 expects tensors with a size of N x 3 x 299 x 299,
    so ensure your images are sized accordingly.
    For torchvision's implementation reason, currently we can not get 4-dim features from pratrained InceptionV3 model
    without modifying InceptionV3's source code,
    Usage:
        >>> expected_feat_size = [2048,]
        >>> extractor = InceptionV3FeatureExtractor(expected_feat_size)
        >>> features = extractor.extract_features(pil_image)
        >>> assert list(features.shape)==expected_feat_size
    """

    TARGET_IMAGE_SIZE = [299, 299]
    MODEL_ARCH = ["inception_v3"]

    def __init__(self, *args, **kwargs):
        self._model_arch = "inception_v3"
        super().__init__(*args, **kwargs)

    def get_model_name(self):
        return self._model_arch

    def get_feature_name(self):
        return "pool_before_linear"

    def _build_extractor(self):
        net = getattr(models, self._model_arch)(
            transform_input=False, aux_logits=False, pretrained=self._pratrained
        )
        net.fc = (
            torch.nn.Identity()
        )  # noqa: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119
        net.eval()
        # Note: torch.vision's InceptionV3 implementation has torch.nn.functional,
        # module.children only return modules, functional ops(F.adaptive_avg_pool2d, F.dropout, F.max_pool2d)
        # will not appear in reconstruction graph
        # following implementation is wrong:
        # modules = list(net.children())[:-1]
        # net = torch.nn.Sequential(*modules)
        if torch.cuda.is_available():
            net.to("cuda")
        return net

    def _build_postprocessor(self):
        return lambda x: x[0]
