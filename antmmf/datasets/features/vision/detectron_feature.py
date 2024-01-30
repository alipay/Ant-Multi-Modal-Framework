# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import cv2
import numpy as np
import torch

from antmmf.common.registry import registry
from antmmf.datasets.features.vision.base_extractor import OnlineFeatureExtractor

__all__ = ["DetectronFeatureExtractor", "Detectron2FeatureExtractor"]


class DetectronFeatureExtractor(OnlineFeatureExtractor):
    """
    Requires vqa-maskrcnn-benchmark to be built and installed Category mapping for visual genome can be downloaded from
    https://dl.fbaipublicfiles.com/pythia/data/visual_genome_categories.json
    currently using detectron model and config can be found at:
    MODEL_URL = [
        "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth"
    ]
    CONFIG_URL = [
        "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml"
    ]
    see issue at: https://github.com/facebookresearch/mmf/issues/100
    """

    MAX_SIZE = 1333
    MIN_SIZE = 800
    CHANNEL_MEAN = [102.9801, 115.9465, 122.7717]  # BGR format

    def __init__(
        self,
        detectron_config_file,
        detectron_model_file,
        feature_name="fc6",
        *args,
        **kwargs,
    ):
        self.writer = registry.get("writer")
        self._detectron_config_file = detectron_config_file
        self._detectron_model_file = detectron_model_file
        self._feature_name = feature_name
        super().__init__(*args, **kwargs)

    def get_model_name(self):
        return "FasterRCNN-101-FPN"

    def get_feature_name(self):
        return self._feature_name

    def _build_preprocessor(self):
        def wrapper(pil_img):
            im = np.array(pil_img).astype(np.float32)

            if im.shape[-1] > 3:
                im = np.array(pil_img.convert("RGB")).astype(np.float32)

            # IndexError: too many indices for array, grayscale images
            if len(im.shape) < 3:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)

            im = im[:, :, ::-1]  # rgb -> bgr
            im -= np.array(DetectronFeatureExtractor.CHANNEL_MEAN)
            im_height, im_width, channels = im.shape
            im_size_min = np.min([im_height, im_width])
            im_size_max = np.max([im_height, im_width])

            # Scale based on minimum size
            im_scale = self.__class__.MIN_SIZE / im_size_min

            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.__class__.MAX_SIZE:
                im_scale = self.__class__.MAX_SIZE / im_size_max

            im = cv2.resize(
                im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
            )  # BGR

            img = torch.from_numpy(im).permute(2, 0, 1)  # channel_frist BGR image

            im_info = {"width": im_width, "height": im_height}

            return img, im_scale, im_info

        return wrapper

    def extract_features(self, pil_imgs):
        """
        Args:
            pil_img(list): list of RGB mode PIL.Image object, needed to be channels_first

        Returns:
            feature(list)： list of np.ndArray, with each item a 2-dim float32 array of shape (N, 2048) , 2048-dim
            feature for each box
            feature_info(list): list of dict, with each item a dict containing box position & class info

        """
        if not isinstance(pil_imgs, (list, tuple)):  # support batch inference
            pil_imgs = [pil_imgs]
        img_tensor, im_scales, im_infos = [], [], []
        for pil_img in pil_imgs:
            im, im_scale, im_info = self._preprocessor(pil_img)
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)
        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        from maskrcnn_benchmark.structures.image_list import to_image_list

        current_img_list = to_image_list(img_tensor, size_divisible=32)
        if torch.cuda.is_available():
            current_img_list = current_img_list.to("cuda")
        with torch.no_grad():
            # TODO: support feature extraction for input boxes
            # refer commit:
            # https://github.com/ronghanghu/vqa-maskrcnn-benchmark-m4c
            output = self._extractor(current_img_list)
            # _postprocessor supports batch_inference
            features, feature_infos = self._postprocessor(output, im_scales, im_infos)
        return features, feature_infos

    def _build_extractor(self):
        from maskrcnn_benchmark.modeling.detector import build_detection_model
        from maskrcnn_benchmark.config import cfg

        cfg.merge_from_file(self._detectron_config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        if self._detectron_model_file:
            if self.writer is not None:
                self.writer.write(
                    f"loaded model {self._detectron_model_file} for detectron feature"
                )
            checkpoint = torch.load(
                self._detectron_model_file, map_location=torch.device("cpu")
            )
            from maskrcnn_benchmark.utils.model_serialization import load_state_dict

            load_state_dict(model, checkpoint.pop("model"))
        else:
            warnings.warn("no model file is assigned to extract detectron feture")
        if torch.cuda.is_available():
            model.to("cuda")
        model.eval()
        return model

    def _build_postprocessor(self):
        def wrapper(
            output,
            im_scales,
            im_infos,
            num_features=100,
            conf_thresh=0.2,
            start_index=1,
        ):
            from maskrcnn_benchmark.layers import nms

            batch_size = len(output[0]["proposals"])
            n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
            score_list = output[0]["scores"].split(n_boxes_per_image)
            score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
            feats = output[0][self._feature_name].split(n_boxes_per_image)
            cur_device = score_list[0].device

            feat_list = []
            info_list = []

            for i in range(
                batch_size
            ):  # for online inference, batch_size should always be 1
                dets = output[0]["proposals"][i].bbox / im_scales[i]
                scores = score_list[i]
                max_conf = torch.zeros(scores.shape[0]).to(cur_device)
                conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
                # Column 0 of the scores matrix is for the background class
                for cls_ind in range(start_index, scores.shape[1]):
                    cls_scores = scores[:, cls_ind]
                    keep = nms(dets, cls_scores, 0.5)
                    max_conf[keep] = torch.where(
                        # Better than max one till now and minimally greater
                        # than conf_thresh
                        (cls_scores[keep] > max_conf[keep])
                        & (cls_scores[keep] > conf_thresh_tensor[keep]),
                        cls_scores[keep],
                        max_conf[keep],
                    )

                sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
                num_boxes = (sorted_scores[:num_features] != 0).sum()
                keep_boxes = sorted_indices[:num_features]
                feat_list.append(feats[i][keep_boxes].cpu().numpy())
                bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
                # Predict the class label using the scores
                objects = torch.argmax(scores[keep_boxes][:, start_index:], dim=1)

                info_list.append(
                    {
                        "bbox": bbox.cpu().numpy(),
                        "num_boxes": num_boxes.item(),
                        "objects": objects.cpu().numpy(),
                        "cls_prob": scores[keep_boxes][:, start_index:].cpu().numpy(),
                        "image_width": im_infos[i]["width"],
                        "image_height": im_infos[i]["height"],
                    }
                )

            return feat_list, info_list

        return wrapper

    @staticmethod
    def vis_feature_info(pil_image, feature_info, visual_genome_categories_json=None):
        """
        visualization func for feature_info
        Args:
            pil_image: RGB mode PIL.Image object, needed to be channels_first
            feature_info: prediction result by func `DetectronFeatureExtractor.extract_features`
            visual_genome_categories_json: visual_genome dataset catergories, can be downloaded from
            https://dl.fbaipublicfiles.com/pythia/data/visual_genome_categories.json

        Returns:
        Usage:
            >>> kModelPath = '~/AntAI-MMF/data/model_data'
            >>> detectron_config_file = osp.join(osp.expanduser(kModelPath), 'detectron_model.yaml')
            >>> detectron_model_file = osp.join(osp.expanduser(kModelPath), 'detectron_model.pth')
            >>> cates_info_json = osp.join(osp.expanduser(kModelPath), 'visual_genome_categories.json')
            >>> feature_extractor = DetectronFeatureExtractor(detectron_config_file, detectron_model_file)
            >>> features, feature_infos = feature_extractor.extract_features([pil_image])
            >>> DetectronFeatureExtractor.vis_feature_info(pil_image, feature_infos[0], cates_info_json)
        """
        idx2cate_map = None
        if visual_genome_categories_json is not None:
            import json
            import codecs

            cates_info = json.load(
                codecs.open(visual_genome_categories_json, "r", "utf-8")
            )
            idx2cate_map = dict(
                [
                    (cate_info["id"], cate_info["name"])
                    for cate_info in cates_info["categories"]
                ]
            )

        bbox, objs = (
            np.rint(feature_info["bbox"]),
            np.int32(feature_info["objects"]),
        )

        if "cls_prob" in feature_info:
            cls_probs = np.float32(feature_info["cls_prob"])
        else:
            cls_probs = np.ones((bbox.shape[0],))

        from antmmf.utils.visual_utils import Box, draw

        num_box = bbox.shape[0]
        box_list = []
        for i in range(num_box):
            cls_ind = objs[i]
            cls_prob = cls_probs[i].max()
            text = "%s:%.2f" % (
                idx2cate_map[cls_ind] if idx2cate_map is not None else cls_ind,
                cls_prob,
            )
            box_list.append(Box(bndbox=bbox[i], text=text))
        vt = draw(np.array(pil_image)[:, :, ::-1], box_list, color=(0, 255, 0))

        cv2.imshow("vis_feature_info", vt.canvas)
        cv2.waitKey()


class Detectron2FeatureExtractor(DetectronFeatureExtractor):
    """
    We modified this repository based on Facebook detectron2 v0.2.1.
    """

    def _build_preprocessor(self):
        def wrapper(pil_img):
            im = np.array(pil_img).astype(np.float32)
            img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            return img

        return wrapper

    def _add_detectron_config(self, cfg):
        cfg.DATASETS.VIS_DIR = ""
        cfg.DATASETS.DATASET_DIR = ""

    def _build_extractor(self):
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        self._add_detectron_config(cfg)
        cfg.merge_from_file(self._detectron_config_file)
        if self._detectron_model_file:
            cfg.MODEL.WEIGHTS = self._detectron_model_file

        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"

        # following volta config:
        # https://github.com/e-bug/volta/blob/main/data/conceptual_captions/extract_cc_image.py
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2

        cfg.freeze()
        model = DefaultPredictor(cfg)
        return model

    def _build_postprocessor(self):
        def wrapper(
            output,
            max_num_boxes=100,
        ):
            detection_result = output["instances"]
            feature = (
                detection_result.features[:max_num_boxes].cpu().numpy()
            )  # num_box, feat_dim

            info = {
                "bbox": detection_result.pred_boxes.tensor[:max_num_boxes]
                .cpu()
                .numpy(),
                "num_boxes": min(len(detection_result), max_num_boxes),
                "objects": detection_result.pred_classes[:max_num_boxes]
                .cpu()
                .numpy(),  # cls label
                "cls_prob": detection_result.class_probs[:max_num_boxes]
                .cpu()
                .numpy(),  # cls distribution
                "image_width": detection_result.image_size[1],
                "image_height": detection_result.image_size[0],
            }

            return feature, info

        return wrapper

    def extract_features(self, pil_imgs):
        """
        Args:
            pil_img(list): list of RGB mode PIL.Image object, needed to be channels_first

        Returns:
            feature(list)： list of np.ndArray, with each item a 2-dim float32 array of shape (N, 2048) , 2048-dim
            feature for each box
            feature_info(list): list of dict, with each item a dict containing box position & class info

        """
        if not isinstance(pil_imgs, (list, tuple)):  # support batch inference
            pil_imgs = [pil_imgs]
        features, feature_infos = [], []
        for pil_img in pil_imgs:
            im = self._preprocessor(pil_img)
            output = self._extractor(im, return_roi_feature=True)
            feature, feature_info = self._postprocessor(output)
            features += [feature]
            feature_infos += [feature_info]
        return features, feature_infos
