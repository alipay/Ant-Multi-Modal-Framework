# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import collections
import os.path as osp

import numpy as np
import torch

from antmmf.common import constants
from antmmf.common import Configuration
from antmmf.common.registry import registry
from antmmf.structures import NestedTensor
from antmmf.structures import Sample
from antmmf.structures import SampleList
from antmmf.datasets.base_dataset import BaseDataset
from antmmf.utils.general import check_required_keys
from antmmf.utils.general import get_antmmf_root
from antmmf.structures import Boxes
from ...roi.dataset import (
    AddITMMixin,
    AddCaptionMixin,
    AddGenerationMixin,
    AddOCRMixin,
    RoiJsonAnnotated,
    ImageM2OCRDataset,
)


class AddDetectionObjs(object):
    def add_ocr_bbox(self, image_info_0, sample_info, ocr_box_label):
        ocr_bbox = np.array(sample_info["line_bboxes"]).reshape([-1, 4])
        ocr_info = dict(
            bbox=ocr_bbox, objects=ocr_box_label * np.ones([ocr_bbox.shape[0]])
        )
        num_box_ocr = ocr_bbox.shape[0]
        if isinstance(image_info_0, collections.abc.Mapping) and check_required_keys(
            image_info_0, ["bbox", "objects"]
        ):
            ocr_info["bbox"] = np.concatenate(
                [image_info_0["bbox"], ocr_info["bbox"]], 0
            )
            assert ocr_box_label not in image_info_0["objects"].tolist(), image_info_0[
                "objects"
            ]
            ocr_info["objects"] = np.concatenate(
                [image_info_0["objects"], ocr_info["objects"]], 0
            )
            num_box_obj = image_info_0["objects"].shape[0]
        else:
            num_box_obj = 0

        return ocr_info, [num_box_obj, num_box_ocr]

    def add_detr_image_and_objs(
        self, sample_info, sample, ocr_box_label=15, topk_detect_objs=15
    ):
        assert check_required_keys(sample_info, ["image", "image_info_0"])
        image_info_0 = sample_info["image_info_0"]
        sample.image_mask = torch.from_numpy(
            sample_info["image_info"]["image_mask"]
        ).clone()
        image_size = [
            sample_info["image_info"]["image_height"],
            sample_info["image_info"]["image_width"],
        ]
        sample.image_raw_size = torch.as_tensor(image_size, dtype=torch.long)
        sample.image_name = sample_info["image_name"]
        sample.image_abs_path = sample_info["image_path"]

        # add ocr boxes to transform with obj boxes.
        image_info_0, [num_box_obj, num_box_ocr] = self.add_ocr_bbox(
            image_info_0, sample_info, ocr_box_label
        )
        # from antmmf.datasets.features.vision.detectron_feature import DetectronFeatureExtractor
        # DetectronFeatureExtractor.vis_feature_info(sample_info['image'][0], image_info_0)
        boxes_idx = torch.cat(
            [-torch.ones([num_box_obj]).long(), torch.arange(num_box_ocr)], 0
        )

        # OCR需要和图像(random resize)匹配，做相同的变换
        # transform image with all boxes
        result = self.detr_processor(
            {"image": sample_info["image"][0], "target": image_info_0}
        )

        # split box into obj bbox & ocr bboxes, Note: bbox is cxcywh now!
        labels, boxes, keep = (
            result["target"]["labels"],
            result["target"]["boxes"],
            result["target"]["keep"],
        )

        # save remained box index
        keep_ocr_boxes_idx = torch.tensor(
            [x for x in boxes_idx[keep].tolist() if x != -1], dtype=torch.long
        )
        remove_ocr_boxes_idx = torch.tensor(
            [x for x in boxes_idx[~keep].tolist() if x != -1], dtype=torch.long
        )
        ocr_line_boxes = boxes.new_zeros([num_box_ocr, 4])
        ocr_line_labels = labels.new_zeros([num_box_ocr], dtype=torch.long)
        ocr_line_boxes[keep_ocr_boxes_idx] = boxes[labels == ocr_box_label]
        ocr_line_labels[keep_ocr_boxes_idx] = labels[labels == ocr_box_label]

        ocr_valid = [
            (idx, ocr_box_idx)
            for idx, ocr_box_idx in enumerate(sample_info["ocr_box_idx"])
            if ocr_box_idx not in remove_ocr_boxes_idx
        ]
        sample_info["ocr_text"] = [sample_info["ocr_text"][x[0]] for x in ocr_valid]
        sample_info["ocr_box"] = (
            (
                Boxes(
                    ocr_line_boxes[
                        torch.tensor([x[1] for x in ocr_valid], dtype=torch.long)
                    ],
                    Boxes.BOX_MODE_CXCYWH,
                )
                .convert_box_mode(Boxes.BOX_MODE_XYXY)
                .tensor
                * 1000
            )
            .long()
            .tolist()
        )  # x1y1x2y2 range: [0, 1000], the same with original ocr_box inputs.

        # from antmmf.datasets.features.vision.detectron_feature import DetectronFeatureExtractor
        # from PIL import Image
        # vis_img = (result["image"].permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor(
        #     [0.485, 0.456, 0.406])) * 255
        # vis_img = Image.fromarray(vis_img.to(torch.uint8).numpy())
        # img_w, img_h = vis_img.size
        # vis_boxes = torch.tensor(sample_info["ocr_box"])/1000*torch.tensor([img_w, img_h, img_w, img_h])
        # DetectronFeatureExtractor.vis_feature_info(vis_img, {"bbox": vis_boxes.numpy(),
        #                                                              "objects": np.ones(vis_boxes.shape[0])})

        # step2: use all bbox for detection
        detect_box = boxes
        detect_label = labels

        # sort by boxes' area, bbox is cxcywh format
        cx, cy, w, h = np.split(detect_box, 4, axis=1)
        area = (w * h)[:, 0]
        idx = np.argsort(-area)[:topk_detect_objs]
        detect_box, detect_label = detect_box[idx], detect_label[idx]

        # pad to 100
        pad_target = self.detr_processor.pad_target(
            {"boxes": detect_box, "labels": detect_label}, num_box_max=100
        )

        sample.image_data = torch.stack([result["image"]], 0)

        sample.obj_target = Sample()
        sample.obj_target.orig_size = result["target"]["orig_size"]  # 2,
        sample.obj_target.size = result["target"]["size"]  # 2,

        sample.obj_target.boxes = pad_target["boxes"]  # 100, 4
        sample.obj_target.labels = pad_target["labels"]  # 100,
        sample.obj_target.num_box = pad_target["num_box"]  # 1,


class MMFUnivlDataset(
    BaseDataset,
    AddDetectionObjs,
    AddITMMixin,
    AddOCRMixin,
    AddCaptionMixin,
    AddGenerationMixin,
):

    NAME = "univl_dataset"

    def __init__(self, dataset_type, config):
        super().__init__(self.__class__.NAME, dataset_type, config)

    def setup_extras(self, dataset_type, config, *args, **kwargs):
        self.detr_processor = (
            self.train_detr_processor
            if dataset_type == "train"
            else self.test_detr_processor
        )
        # No transform for ImageDataset, image and bbox should
        # transform together for image with bbox annotations.
        # self.image_db.transform = self.image_processor
        # assign annotation_db to image_db
        self.image_db.annotation_db = self.annotation_db

        # assign processor, for tokenizing ocr tokens
        self.image_db.tokenizer = getattr(self.ocr_processor, "_tokenizer")

    def __len__(self):
        return len(self.image_db)

    def get_item(self, idx):
        sample_info = self.image_db[idx]
        if sample_info is None:
            return None

        if self.config.get(constants.USE_FEATURE_STR, False):
            # load bbox annotations
            feat_info = self.feature_db[idx]
            sample_info.update(feat_info)

        # bbox visualization
        # from antmmf.datasets.features.vision.detectron_feature import DetectronFeatureExtractor
        # DetectronFeatureExtractor.vis_feature_info(sample_info['image'][0], feat_info['image_info_0'])

        current_sample = Sample()

        # step1： image
        self.add_detr_image_and_objs(sample_info, current_sample, ocr_box_label=15)

        # step2: random ocr and caption before process
        # Note: typically we only randomly replace captions, here we randomly replace both
        # for rare existing of caption field in business data.
        self.add_itm_label(
            sample_info, current_sample, idx, random_ocr=True, random_caption=True
        )

        # step3: ocr
        self.add_ocr_details(sample_info, current_sample)

        # step4: caption
        self.add_caption(sample_info, current_sample)

        # step5: add generation
        self.add_generation(sample_info, current_sample)

        # step5: 增加label
        if sample_info.get("label") is not None:
            current_sample["targets"] = torch.tensor(
                int(sample_info["label"]), dtype=torch.long
            )

        return Sample(current_sample)

    def collate_fn(self, batch):
        # filter None sample
        batch = [x for x in batch if x is not None]
        # each sample only have one image
        tensor_list = [sample.image_data[0] for sample in batch]
        # 这里的padding不改变 obj bbox & ocr bbox 的坐标
        pad_imgs, pad_masks = NestedTensor.from_tensor_list(tensor_list).decompose()
        for i in range(len(batch)):
            batch[i].image_data = pad_imgs[i][None]
            batch[i].image_pad_mask = pad_masks[i][None]

        samplelist = SampleList(batch)
        return samplelist

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
    test_yaml = "configs/univl/univl_pretrain.yml"
    config_yaml_file = osp.join(get_antmmf_root(), "..", test_yaml)
    config = Configuration(config_yaml_file)
    config.freeze()
    dataset_config = config.task_attributes.univl_task.dataset_attributes.univl_dataset
    from antmmf.utils.logger import Logger

    registry.register("writer", Logger(config))

    train_dataset = MMFUnivlDataset("test", dataset_config)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=train_dataset.collate_fn,
    )

    for i_batch, batched in enumerate(train_loader):
        print(batched)
