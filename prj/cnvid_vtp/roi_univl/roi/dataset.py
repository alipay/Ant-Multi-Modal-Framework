# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import codecs
import copy
import json
import os.path as osp
import random
import warnings
from json.decoder import JSONDecodeError
from typing import Optional

import numpy as np
import torch

from antmmf.common import Configuration
from antmmf.common import constants
from antmmf.common.registry import registry
from antmmf.datasets.base_dataset import BaseDataset
from antmmf.datasets.database.annotated import AnnotatedDatabase
from antmmf.datasets.database.image_database import ImageDatabase
from antmmf.structures import NestedTensor
from antmmf.structures import Sample
from antmmf.utils.file_io import PathManager
from antmmf.utils.general import check_required_keys
from antmmf.utils.general import get_antmmf_root
from antmmf.utils.image_ops import ExifImageLoader


# analyze annotations
class RoiJsonAnnotated(AnnotatedDatabase):
    def __init__(self, annotation_file, dataset_instance: Optional[BaseDataset] = None):
        self.index_file = annotation_file
        self.dataset_instance = dataset_instance
        self.config = dataset_instance.config
        self._dataset_type = dataset_instance._dataset_type
        self.writer = registry.get("writer")
        super().__init__(annotation_file)
        self._parse_path_from_config()

    def _parse_path_from_config(self):
        image_dir, ocr_dir, feat_dir = None, None, None
        if self.config.get("use_images", False):
            image_dir = self.dataset_instance._get_absolute_path(
                self.config["images"][self._dataset_type]
            )
        if self.config.get(constants.USE_FEATURE_STR, False):
            feat_dir = self.dataset_instance._get_absolute_path(
                self.config[constants.FEATURES_STR][self._dataset_type]
            )
        if self.config.get("use_ocrs", False):
            ocr_dir = self.dataset_instance._get_absolute_path(
                self.config["ocrs"][self._dataset_type]
            )
        self.image_dir = image_dir
        self.ocr_dir = ocr_dir
        self.feat_dir = feat_dir

    def preprocess_item(self, record):
        data_item = {}
        data_item["caption"] = record.get("caption", "")

        if self.image_dir is not None:
            data_item["image_path"] = osp.join(self.image_dir, record["image"])
        if self.ocr_dir is not None:
            data_item["ocr_path"] = osp.join(self.ocr_dir, record["ocr"])
        if self.feat_dir is not None:
            data_item[constants.FEATURE_PATH_STR] = osp.join(
                self.feat_dir, record["feature"]
            )
            data_item["feat_info"] = osp.join(
                self.feat_dir, record["feature"].split(".npy")[0] + "_info.npy"
            )
        data_item["label"] = record.get("label")
        return data_item


# load image and apply transforms
class ImageDataset(ImageDatabase):
    def __init__(
        self,
        imgdir,
        annotation_db,
        transform=None,
        loader=ExifImageLoader.load_with_exif,
    ):
        super().__init__(imgdir, annotation_db, transform, loader=loader)

    def get(self, annotation_item):
        annotation_item = copy.deepcopy(annotation_item)
        # image input
        image_info = super().get(annotation_item)

        # allow one image input
        assert len(image_info["images"]) == 1
        if image_info["images"][0] is None:
            return None

        # image_data -> image
        annotation_item["image"] = image_info["images"]  # N,num_images, 3,224,224
        annotation_item["image_name"] = osp.basename(annotation_item["image_path"])
        # other image_related -> image_info
        annotation_item["image_info"] = Sample()
        annotation_item["image_info"]["image_mask"] = image_info["images_mask"]
        annotation_item["image_info"]["image_height"] = image_info["image_shape"][0][0]
        annotation_item["image_info"]["image_width"] = image_info["image_shape"][0][1]
        return annotation_item


class OCRDataset(object):
    def _load_ocr_json(self, cbox_path, image_height, image_width):
        token_text_buffer, token_bbox_buffer, token_bbox_line_idx, json_res = (
            [],
            [],
            [],
            {},
        )
        line_text_buffer, line_bbox_buffer = [], []  # ocr detection bbox
        if PathManager.isfile(cbox_path):
            try:
                json_res = json.load(codecs.open(cbox_path, "r", "utf-8"))
            except JSONDecodeError:  # corrupted json file
                warnings.warn(f"Corrupted json file:{cbox_path}")
                json_res = {}
        ocr_list = json_res.get("data", {}).get("result", {}).get("result", [])
        for ocr_box in ocr_list:
            # refer to layoutlmtokenizer:
            # https://huggingface.co/transformers/model_doc/layoutlm.html#layoutlmtokenizer

            bbox = np.array(ocr_box["bndbox"])
            # handle into x1y1x2y2 format
            xmin, xmax = min(bbox[:, 0]), max(bbox[:, 0])
            ymin, ymax = min(bbox[:, 1]), max(bbox[:, 1])
            # line box is not normalized
            line_bbox_buffer.append([xmin, ymin, xmax, ymax])

            # char_bbox uses the line-box instead, normalized by image_height, image_width and
            # rescale to range [0, 1000]
            normalized_word_boxes = [
                int(min(max(x, 0.0), 1000))
                for x in [
                    xmin * 1000.0 / image_width,
                    ymin * 1000.0 / image_height,
                    xmax * 1000.0 / image_width,
                    ymax * 1000.0 / image_height,
                ]
            ]

            text = ocr_box["content"]
            line_text_buffer.append(text)
            # word level, not char level
            text_words = self.tokenizer.convert_ids_to_tokens(
                self.tokenizer(text)["input_ids"], skip_special_tokens=True
            )

            token_text_buffer.extend(text_words)
            token_bbox_buffer.extend([normalized_word_boxes] * len(text_words))
            token_bbox_line_idx.extend([len(line_bbox_buffer) - 1] * len(text_words))

            # add separation for each ocr bbox
            token_text_buffer.append(";")
            token_bbox_buffer.append(normalized_word_boxes)
            token_bbox_line_idx.append(len(line_bbox_buffer) - 1)
        return (
            token_text_buffer,
            token_bbox_buffer,
            line_text_buffer,
            line_bbox_buffer,
            token_bbox_line_idx,
        )

    def load_ocr_info(self, annotation_item):
        # load ocrs
        assert check_required_keys(annotation_item, ["ocr_path"])
        if "ocr_path" in annotation_item:
            height = annotation_item["image_info"]["image_height"]
            width = annotation_item["image_info"]["image_width"]

            (
                texts,
                bboxes,
                line_texts,
                line_bboxes,
                token_bbox_line_idx,
            ) = self._load_ocr_json(annotation_item["ocr_path"], height, width)
            annotation_item["ocr_text"] = texts
            annotation_item["ocr_box"] = bboxes
            annotation_item["ocr_box_idx"] = token_bbox_line_idx
            annotation_item["line_texts"] = line_texts
            annotation_item["line_bboxes"] = line_bboxes

        return annotation_item


class ImageM2OCRDataset(ImageDataset, OCRDataset):
    def get(self, annotation_item):
        annotation_item = super().get(annotation_item)
        if annotation_item is not None:  # image not exists, skip loading ocr info
            annotation_item = self.load_ocr_info(annotation_item)
        return annotation_item


class AddITMMixin(object):
    def get_rand_index(self, current_sample_idx):
        rand_caption_idx = current_sample_idx
        if random.random() > 0.5:
            # TODO: add the hard negative mining objective here.
            rand_ocr_info = None
            while (
                rand_caption_idx == current_sample_idx or rand_ocr_info is None
            ):  # in case of negative images that are not exist.
                rand_caption_idx = random.randint(0, len(self.image_db) - 1)
                # image db may return None
                rand_ocr_info = self.image_db[rand_caption_idx]
        return rand_caption_idx

    def _sample_from_ocr_db(self, index):
        ocr_text = self.image_db[index]["ocr_text"]
        ocr_box = self.image_db[index]["ocr_box"]
        return ocr_text, ocr_box

    def _sample_from_caption_db(self, index):
        return self.annotation_db[index]["caption"]

    def add_itm_label(
        self,
        sample_info,
        sample,
        current_sample_idx,
        random_ocr=True,
        random_caption=True,
    ):
        itm_label = 1
        sample_info["origin_caption"] = copy.deepcopy(sample_info["caption"])
        if self.config.pretraining is True and (random_ocr or random_caption):
            rand_idx = self.get_rand_index(current_sample_idx)
            replace = rand_idx != current_sample_idx
            if replace:
                itm_label = 0
            if random_ocr and replace:
                assert check_required_keys(sample_info, ["ocr_text", "ocr_box"])
                ocr_text, ocr_box = self._sample_from_ocr_db(rand_idx)
                sample_info["ocr_text"] = ocr_text
                sample_info["ocr_box"] = ocr_box
            if random_caption and replace:
                assert check_required_keys(sample_info, ["caption"])
                sample_info["caption"] = self._sample_from_caption_db(rand_idx)
        sample.itm_label = torch.tensor(itm_label, dtype=torch.long)


class AddCaptionMixin(object):
    def add_caption(self, sample_info, sample):
        assert hasattr(self, "caption_processor")
        if "caption" in sample_info:
            # add caption for input
            caption_input = self.caption_processor({"text": sample_info["caption"]})
            for key, val in caption_input.items():
                sample["caption_" + key] = val
            caption_raw_input = self.caption_processor(
                {"text": sample_info["caption"]}, probability=0.0
            )
            sample["caption_raw_input_ids"] = caption_raw_input["input_ids"]
        if "tid" in sample_info:
            sample.caption_tid = torch.tensor(sample_info["tid"], dtype=torch.long)
        if "t2vid_list" in sample_info:
            sample.caption_vid_list = sample_info["t2vid_list"]
        if self.config.get("add_false_caption", False):
            max_try = 3
            text_false = None
            while max_try > 0 and text_false is None:
                rand_idx = random.randint(0, len(self.annotation_db) - 1)
                if (
                    self.annotation_db[rand_idx]["video_idx"]
                    != sample_info["video_idx"]
                ):  # not aligned pair
                    if self.annotation_db[rand_idx]["caption"] is not None:
                        text_false = self.annotation_db[rand_idx]["caption"]
                max_try -= 1
            if text_false is None:
                text_false = "this is a dummy text"
            caption_false = self.caption_processor(
                {"text": text_false}, probability=0.0
            )
            sample["caption_false_input_ids"] = caption_false["input_ids"]
            sample["caption_false_input_mask"] = caption_false["input_mask"]
            sample["caption_false_text"] = caption_false["text"]


class AddGenerationMixin(object):
    def add_generation(self, sample_info, sample):
        assert hasattr(self, "caption_processor")
        if "origin_caption" in sample_info:
            # add caption for generation, not masked
            caption_output = self.caption_processor(
                {"text": sample_info["origin_caption"]}, probability=0.0
            )
            for key, val in caption_output.items():
                sample["generation_" + key] = val


class AddImageMixin(object):
    def add_image(self, sample_info, sample):
        if not self.config.get("use_images", False):
            return
        assert check_required_keys(sample_info, ["image", "image_info"])
        assert check_required_keys(sample_info["image_info"], ["image_mask"])

        sample.image_data = torch.stack(
            sample_info["image"], 0
        )  # N,num_images, 3,224,224
        sample.image_mask = torch.from_numpy(
            sample_info["image_info"]["image_mask"]
        ).clone()


class AddOCRMixin(object):
    def add_ocr_details(self, sample_info, sample):
        if not self.config.get("use_ocrs", False):
            return
        assert check_required_keys(sample_info, ["ocr_text", "ocr_box"]) and hasattr(
            self, "ocr_processor"
        )

        ocr_output = self.ocr_processor(
            {"text": sample_info["ocr_text"], "bbox": sample_info["ocr_box"]}
        )
        for key, val in ocr_output.items():
            # # ignore all lm labels in case of Images & Region and OCR are not match
            # if key == constants.LM_LABEL_IDS_STR and itm_label == 0:
            #     # itm_label can only be set as 0 during pre-training phase
            #     val = torch.zeros_like(val).fill_(-1)
            sample["ocr_" + key] = val


class AddRegionMixin(object):
    def add_region_info(self, sample_info, sample):
        if not self.config.get(constants.USE_FEATURE_STR, False):
            return
        assert check_required_keys(sample_info, ["image_info_0", "image_feature_0"])
        assert hasattr(self, "region_processor")
        region_output = self.region_processor(
            {
                "image_info_0": sample_info["image_info_0"],
                "image_feature_0": sample_info["image_feature_0"],
            }
        )
        sample.update(region_output)
        return sample


class MMFRoiDataset(
    BaseDataset,
    AddITMMixin,
    AddCaptionMixin,
    AddGenerationMixin,
    AddImageMixin,
    AddOCRMixin,
    AddRegionMixin,
):
    """
    dataset for Region & OCR & Image joint modelling
    """

    NAME = "roi_dataset"

    def __init__(self, dataset_type, config):
        super().__init__(self.__class__.NAME, dataset_type, config)

    def setup_extras(self, dataset_type, config, *args, **kwargs):
        self.image_processor = (
            self.train_image_processor
            if dataset_type == "train"
            else self.test_image_processor
        )
        # assign transform processor to image_db
        self.image_db.transform = self.image_processor
        # assign annotation_db to image_db
        self.image_db.annotation_db = self.annotation_db
        # assign processor
        self.image_db.tokenizer = getattr(self.ocr_processor, "_tokenizer")

    def __len__(self):
        return len(self.image_db)

    def get_item(self, idx):
        sample_info = self.image_db[idx]
        if sample_info is None:
            return None

        if self.config.get(constants.USE_FEATURE_STR, False):
            feat_info = self.feature_db[idx]
            sample_info.update(feat_info)

        current_sample = Sample()

        # step1： image
        self.add_image(sample_info, current_sample)

        # step2: random ocr and caption before process
        # save origin_caption for caption generation
        self.add_itm_label(
            sample_info, current_sample, idx, random_ocr=True, random_caption=True
        )

        # step3: ocr
        self.add_ocr_details(sample_info, current_sample)

        # step4: caption
        self.add_caption(sample_info, current_sample)

        # step5: region
        self.add_region_info(sample_info, current_sample)

        # step5: 增加label
        if sample_info.get("label") is not None:
            current_sample["targets"] = torch.tensor(
                int(sample_info["label"]), dtype=torch.long
            )

        return Sample(current_sample)

    def collate_fn(self, batch):
        # each sample only have one image
        tensor_list = [sample.image_data[0] for sample in batch]
        pad_imgs, pad_masks = NestedTensor.from_tensor_list(tensor_list).decompose()
        for i in range(len(batch)):
            batch[i].image_data = pad_imgs[i][None]
            batch[i].image_pad_mask = pad_masks[i][None]
        return batch

    def _build_annotation_db(self):
        return super()._build_annotation_db(
            database_cls=RoiJsonAnnotated, dataset_instance=self
        )

    def _build_image_db(self):
        return super()._build_image_db(database_cls=ImageM2OCRDataset)

    # disable parent class download method
    def _download_requirement(
        self, config, requirement_key, requirement_variation="defaults"
    ):
        return None


if __name__ == "__main__":
    test_yaml = "configs/roi_modelling/roi_model_pretrain.yml"
    config_yaml_file = osp.join(
        get_antmmf_root(), "../../../../antmmf/tasks", test_yaml
    )
    config = Configuration(config_yaml_file)
    config.freeze()
    dataset_config = config.task_attributes.roi_task.dataset_attributes.roi_dataset
    from antmmf.utils.logger import Logger

    registry.register("writer", Logger(config))

    train_dataset = MMFRoiDataset("test", dataset_config)

    for sample in train_dataset:
        # if '038wQpZfTpQAAAAAAAAAAABkCjV1AA' not in sample.img_path:
        #     continue
        print("----------------------")
        print(sample)
