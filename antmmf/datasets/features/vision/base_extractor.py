# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class OnlineFeatureExtractor(object):
    """
    Defines a typical feature extraction procedure for online inference with antmmf:
    input_pil_image -> preprocessor -> feature_extractor
                    -> postprocessor -> returned feature -> (optional)feature saver
    """

    DEFAULT_INPUT_IMAGE_SIZE = [1333, 800]

    def __init__(self, feature_manager=None, *args, **kwargs):
        self._preprocessor = self._build_preprocessor()
        self._extractor = self._build_extractor()
        self._postprocessor = self._build_postprocessor()
        self._feature_manager = feature_manager
        if feature_manager is not None:
            from antmmf.datasets.features.vision.feature_saver import FeatureManager

            assert isinstance(feature_manager, FeatureManager)
            self._feature_manager.register_extractor(self)

    def get_model_name(self):
        raise NotImplementedError

    def get_feature_name(self):
        raise NotImplementedError

    def _build_preprocessor(self):
        """
        preprocess image before fed into the extractor
        Returns: callable function or object, it should have the following signature::
            preprocessor(pil_img) -> extractor's input
        """
        raise NotImplementedError

    def _build_extractor(self):
        """
        build model for feature extraction
        Returns: torch.nn.module object
        """
        raise NotImplementedError(
            "OnlineFeatureExtractor doesn't implement a _build_extractor method"
        )

    def _build_postprocessor(self):
        """
        some necessary procedures before getting final feature: do shape transform, format conversion .etc
        Returns:

        """
        raise NotImplementedError

    def extract_features(self, pil_img):
        """
        define a extraction pipeline, subclasses should override this function to perform feature extraction
        Args:
            pil_img: RGB mode PIL.Image object, needed to be channels_first
        Returns:
            torch.Tensor
        """
        pass

    def save_feature(self, image_paths, *feats_res):
        """
        storage for saving online extracted feature, such can be used for
        offline training or validation

        Args:
            image_paths(list): list of image_paths
            *feats_res: outputs for func `extract_features`
        """
        assert self._feature_manager is not None
        self._feature_manager.save(image_paths, *feats_res)

    def print_extractor_summary(self):
        from torchsummary import summary

        summary(self._extractor, tuple([3] + self.__class__.DEFAULT_INPUT_IMAGE_SIZE))
