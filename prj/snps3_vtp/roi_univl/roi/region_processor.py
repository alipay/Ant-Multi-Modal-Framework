# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import random
import torch
import numpy as np
from dataclasses import dataclass
from typing import List
from antmmf.common import AntMMFConfig
from antmmf.utils.general import check_required_keys
from antmmf.common.registry import registry
from antmmf.datasets.processors.processors import BaseProcessor
from antmmf.structures import Boxes


@registry.register_processor("region_processor")
class RegionProcessor(BaseProcessor):
    """
    Region Processor for processing output of antmmf.datasets.database.features_database.FeaturesDatabase.
    This processor supports following feature list:
    1. truncate or pad region boxes and their corresponding features to max_features.
    2. random mask regions and their highly overlapped neighbourhoods with probability of mask_region_prob.
    3. normalize region coords with image sizes and add dimension indicating region areas.

    """

    @dataclass
    class Config(AntMMFConfig):
        max_features: int = 10  # max regions
        feature_dim: int = 2048  # feature dim for each image region
        region_kl_fc_dim: int = (
            13  # num_class for masked region classification loss(kl_loss)
        )
        required_feature_info_key: List[str] = (
            "max_features",
            "cls_prob",  # required keys for feature-info numpy file
            "bbox",
            "image_height",
            "image_width",
        )
        mask_region_prob: float = 0.15

    def random_region(self, image_feat, num_boxes, overlaps):
        """
        Implementation refer to:
        https://github.com/e-bug/volta/blob/9e5202141920600d58a9c5c17519ca453795d65d/volta/datasets/concept_cap_dataset.py#L636

        Args:
            image_feat: N, 2048, padded image feature
            num_boxes:  M, `real` number of boxes
            overlaps:   M x M

        Returns:
            image_feat: regions that are masked or not
            should_predict_label: 1 indicate regions that should be predicted
            region_mask: 0 indicate those masked regions, 1 indicate those remained
        """

        N = image_feat.shape[0]
        region_mask = np.zeros((N,))  # (N, )
        region_mask[:num_boxes] = 1
        should_predict_label = -1 * np.ones((N,), dtype=np.int64)

        for i in range(num_boxes):
            prob = random.random()
            # mask token with 15% probability

            if prob < self.config.mask_region_prob:
                prob /= self.config.mask_region_prob

                # 90% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0
                # mask the overlap regions into zeros
                # ignore regions that are padded or with high overlaps
                remain_region_mask = np.pad(
                    overlaps[i] <= 0.4,
                    (0, N - num_boxes),
                    "constant",
                    constant_values=(True, False),
                )
                region_mask = np.logical_and(region_mask, remain_region_mask)

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                should_predict_label[i] = 1
            else:
                # no masking token (will be ignored by loss function later)
                should_predict_label[i] = -1

        return image_feat, should_predict_label, region_mask

    def __call__(self, sample_info):
        """
        Args:
            sample_info(dict): Returned dict from antmmf.datasets.database.features_database.FeaturesDatabase.
                               sample_info should have keys of 'image_info_0' and 'image_feature_0'.
        Returns:
            region_feature(torch.Tensor): [N, feature_dim(N==max_features)]
            region_to_predict(torch.Tensor): [N,] 1 indicate regions that should be predicted, -1 means ignored
            region_mask(torch.Tensor): [N,] masked regions that are highly overlapped.
                                       0 indicate those masked regions, 1 indicate those remained
            region_num(torch.Tensor): [], number of real regions(not count padded regions)
            region_cls(torch.Tensor): [N, region_kl_fc_dim], class distribution of each region.
            region_location(torch.Tensor):[N, 5], normalized x1y1x2y2area.
        """
        # image_feature_0, image_info_0
        image_info = sample_info["image_info_0"]  # dict
        image_feature = sample_info["image_feature_0"]  # N,2048

        # padding to max_boxes for each image
        padded_region_feature = np.zeros(
            (self.config.max_features, self.config.feature_dim), dtype=np.float32
        )
        padded_region_cls = np.zeros(
            (self.config.max_features, self.config.region_kl_fc_dim),
            dtype=np.float32,
        )
        padded_region_location = np.zeros(
            (self.config.max_features, 5), dtype=np.float32
        )
        if (image_feature == 0).all().item() is False and check_required_keys(
            image_info, self.config.required_feature_info_key
        ):  # skip those image_feature are padded or with necessary keys missing
            # detected regions for real
            num_regions = min(image_info["max_features"], self.config.max_features)

            padded_region_feature[:num_regions] = image_feature[:num_regions]
            padded_region_cls[:num_regions] = image_info["cls_prob"][:num_regions]
            padded_region_location[:num_regions, :4] = image_info["bbox"][
                :num_regions
            ]  # x1y1x2y2, un-normalized

            # normalize bbox (to 0 ~ 1)
            padded_region_location[:, [0, 2]] = padded_region_location[
                :, [0, 2]
            ] / float(image_info["image_width"])
            padded_region_location[:, [1, 3]] = padded_region_location[
                :, [1, 3]
            ] / float(image_info["image_height"])

            # add box area
            padded_region_location[:, 4] = (
                padded_region_location[:, 2] - padded_region_location[:, 0]
            ) * (padded_region_location[:, 3] - padded_region_location[:, 1])

            padded_region_location = np.clip(padded_region_location, 0, 1)

            # rescale to [0, 1000], similar to OCR box
            padded_region_location[:, :4] *= 1000

            overlaps = Boxes(image_info["bbox"][:num_regions]).pairwise_iou(
                Boxes(image_info["bbox"][:num_regions])
            )

            region_feat, region_to_predict, region_mask = self.random_region(
                padded_region_feature, num_regions, overlaps
            )
        else:
            num_regions = 0
            region_feat = padded_region_feature
            N = region_feat.shape[0]
            region_to_predict = -1 * np.ones((N,), dtype=np.int64)
            region_mask = np.zeros((N,), dtype=np.int64)

        return_dict = dict()
        return_dict["region_feature"] = torch.from_numpy(region_feat).clone().detach()
        return_dict["region_to_predict"] = (
            torch.from_numpy(region_to_predict).clone().detach()
        )
        return_dict["region_mask"] = (
            torch.from_numpy(region_mask).to(torch.int64).clone().detach()
        )
        return_dict["region_num"] = torch.from_numpy(
            np.array(num_regions, dtype=np.int64)
        )
        return_dict["region_cls"] = torch.from_numpy(padded_region_cls).clone().detach()
        return_dict["region_location"] = (
            torch.from_numpy(padded_region_location).to(torch.long).clone().detach()
        )
        return return_dict
