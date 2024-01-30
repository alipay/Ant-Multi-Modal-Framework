# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import copy
import torch
import os
import warnings
from PIL import Image

from antmmf.datasets.database.annotated import AnnotatedDatabase
from antmmf.datasets.database.video_database import VideoClipsDatabase
from antmmf.datasets.base_dataset import _get_config_path
from antmmf.utils.general import get_absolute_path
from .ret_dataset import MMFUnivlVideoDataset
from torchvision import transforms


class ClassificationAnnotated(AnnotatedDatabase):
    def process_annotation(self, annotation_database):
        # train/val format: {"caption": "xxx", "clip_name": "video9771", "label": 0} or
        # multi-label train/val format: {"caption": "xxx", "clip_name": "video9771", "label": [0,1]}
        return annotation_database


class ImageVideoDatabase(VideoClipsDatabase):
    def __init__(self, image_path, video_path, annotation_db, **kwargs):
        super().__init__(video_path, annotation_db, **kwargs)
        self.image_base_path = get_absolute_path(image_path) if image_path else None
        self.image_transform = transforms.PILToTensor()

    def _load_inflated_image(self, image_path):
        if not os.path.exists(image_path):
            warnings.warn(f"image_path not found:{image_path}")
            return None
        try:  # do not assume images are valid
            image_raw = Image.open(image_path).convert("RGB")
        except BaseException:
            warnings.warn(f"corrupt image:{image_path}")
            return None
        image_tensor = self.image_transform(image_raw).unsqueeze(0)
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        image_tensor = image_tensor.repeat_interleave(
            self.ensemble_n_clips * self.num_frm, dim=0
        )
        return image_tensor

    def get(self, item):
        dtype = item["type"]
        if dtype == "video":
            video_info = super().get(item)
            data = video_info["video"]
            video_mask = video_info["video_mask"]
        else:
            # item is image now
            img_name = item["clip_name"]
            image_path = os.path.join(self.image_base_path, img_name)
            data = self._load_inflated_image(image_path)
            video_mask = None

        return_info = {
            "video": data,  # frame format:rgb, (n_clips*num_frm, C, H, W)
            "n_clips": self.ensemble_n_clips,
            "num_frm": self.num_frm,
            "video_mask": video_mask,
        }
        return return_info


class MMFVideoClassificationDataset(MMFUnivlVideoDataset):

    NAME = "video_text_classification"

    def _build_annotation_db(self):
        return super(MMFUnivlVideoDataset, self)._build_annotation_db(
            database_cls=ClassificationAnnotated
        )

    def _build_video_db(self):
        video_path, image_path, asr_path = ".", ".", None

        if self.config.get("use_videos", False):
            video_path = _get_config_path(
                self.config["videos"], self._dataset_type, self._index
            )
            video_path = self._get_absolute_path(video_path)

        if self.config.get("use_images", False):
            image_path = _get_config_path(
                self.config["images"], self._dataset_type, self._index
            )
            image_path = self._get_absolute_path(image_path)

        return ImageVideoDatabase(
            image_path,
            video_path,
            self.annotation_db,
            asr_path=asr_path,
            dataset_type=self._dataset_type,
            **self.config,
        )

    def get_item(self, idx):
        sample_info = copy.deepcopy(self.annotation_db[idx])
        video_info = self.video_db[idx]

        return self._get_one_item(sample_info, video_info)

    def add_label(self, sample_info, sample):
        label = sample_info.get("label")
        if label is not None:
            if isinstance(label, list):  # support multi-label
                if len(label) > 0:
                    if isinstance(label[0], str):  # list of str
                        label = [self.config.labels.index(l) for l in label]
                    elif isinstance(label[0], int):
                        label = label  # list of int label
                    else:
                        raise Exception(f"unkown label type:{label}")

                else:  # support empty-label for mce
                    label = []
                labels = torch.tensor(label, dtype=torch.long).unsqueeze(0)
                num_labels = len(self.config.labels)
                sample["targets"] = (
                    torch.zeros(labels.size(0), num_labels)
                    .scatter_(1, labels, 1.0)
                    .squeeze(0)
                )
            else:
                if isinstance(label, str):  # str label
                    if hasattr(self.config, "labels") and label in self.config.labels:
                        label = self.config.labels.index(label)
                    else:
                        label = int(label)  # num label

                sample["targets"] = torch.tensor(label, dtype=torch.long)

    def format_for_evalai(self, report, thresh=0.0):
        logits = report.logits
        targets = report.targets
        predict = torch.where(logits.sigmoid() >= 0.5, 1, 0).long()
        bsz = targets.size(0)
        batch_results = []
        for idx in range(bsz):
            pred_label = [
                self.config.labels[l] for l in predict[idx].nonzero(as_tuple=True)[0]
            ]
            target_label = [
                self.config.labels[l] for l in targets[idx].nonzero(as_tuple=True)[0]
            ]
            cur_result = {
                "video_id": report.clip_name[idx],
                "pred_label": pred_label,
                "probs": [
                    "%.2f" % x
                    for x in logits[idx][predict[idx].nonzero(as_tuple=True)[0]]
                    .sigmoid()
                    .tolist()
                ],
                "target": target_label,
            }
            batch_results.append(cur_result)
        return batch_results
