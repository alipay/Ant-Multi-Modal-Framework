# Copyright (c) 2023 Ant Group and its affiliates.

import os.path as osp

import numpy as np
import torch

from antmmf.common import constants
from antmmf.common.constants import (
    TEXT_MODALITY,
    IMAGES_STR,
    CLS_ID_STR,
    SEP_ID_STR,
    ID_STR,
)
from antmmf.datasets.base_dataset import BaseDataset
from antmmf.structures import Sample
from antmmf.utils.visualize import visualize_images

"""
This follows Mmbt dataset format
observation: [[CLS] <Image-Feature> [SEP] <Input_IDS>]
position:    [0, 1, ...,  Image-Feature-Len ]
"""


class MmfImageTextDataset(BaseDataset):
    def __init__(self, dataset_name, database_type, config, *args, **kwargs):
        super().__init__(dataset_name, database_type, config, *args, **kwargs)
        assert (
            self.image_db is not None
        ), "config's use_images must be true and images must be set to use image dataset"
        # Assign transforms to the image_db

        # this allows data augmentation during training
        if (
            getattr(self, "train_image_processor", None) is not None
            and getattr(self, "test_image_processor", None) is not None
        ):
            self.image_processor = (
                self.train_image_processor
                if database_type == "train"
                else self.test_image_processor
            )

        # assign transform processor to image_db
        self.image_db.transform = self.image_processor

    @staticmethod
    def build_sample_with_text_image(
        text_sample_info, text_processor, image_sample_info=None, image_processor=None
    ):
        """

        Args:
            text_sample_info(Dict): {'text': ''}
            image_sample_info(Dict): {'images': [image_tensor] }, with image_tensor's shape: [3,h,w]
            text_processor:
            image_processor: optional since images may have been processed in image_db

        Returns:
            sample(amtmmf.structures.sample.Sample):

        """
        current_sample = Sample()

        processed_text = text_processor(
            {TEXT_MODALITY: text_sample_info[TEXT_MODALITY]}
        )
        current_sample.text = processed_text["input_ids"]
        if "token_num" in processed_text:
            current_sample.source_len = processed_text["token_num"]

        if "input_mask" in processed_text:
            current_sample.mask = processed_text["input_mask"]

        # image uses segment = 0,
        # text uses segment = 1, therefore
        if "segment_ids" in processed_text:
            current_sample.segment = processed_text["segment_ids"] + 1

        if ID_STR in text_sample_info:
            current_sample.id = torch.tensor(
                int(text_sample_info[ID_STR]), dtype=torch.int
            )
        if image_sample_info is not None:
            current_sample.image = image_sample_info[IMAGES_STR][0]  # 3,h,w
            if image_processor is not None:
                current_sample.image = image_processor(current_sample.image)
            assert (
                current_sample.image.size()[0] == 3
                and len(current_sample.image.size()) == 3
            ), "need to have [3xhxw] size for image"
            current_sample.image_mask = torch.tensor([1], dtype=torch.long)

        # information about tokenization ids, used in
        # later early fusion between text and image
        if CLS_ID_STR in processed_text:
            current_sample.cls_id = torch.tensor(
                processed_text[CLS_ID_STR], dtype=torch.long
            )
        if SEP_ID_STR in processed_text:
            current_sample.sep_id = torch.tensor(
                processed_text[SEP_ID_STR], dtype=torch.long
            )

        if "label" in text_sample_info:
            current_sample.targets = torch.tensor(
                int(text_sample_info["label"]), dtype=torch.long
            )

        return current_sample

    def get_item(self, idx):
        text_sample_info = self.annotation_db[idx]
        imag_sample_info = self.image_db[idx]

        sample = MmfImageTextDataset.build_sample_with_text_image(
            text_sample_info, self.text_processor, imag_sample_info
        )
        return sample

    def visualize(self, num_samples=1, use_transforms=False, *args, **kwargs):
        image_paths = []
        random_samples = np.random.randint(0, len(self), size=num_samples)

        for idx in random_samples:
            image_paths.append(self.annotation_db[idx]["img"])

        images = self.image_db.from_path(image_paths, use_transforms=use_transforms)
        visualize_images(images[IMAGES_STR], *args, **kwargs)

    def get_annotations(self, attr):
        if attr is constants.IMAGE_NAME_STR:
            return [osp.basename(an["img"]) for an in self.annotation_db]
        else:
            raise NotImplementedError(
                f"Access to attr:{attr} is Not allowed for dataset: {self.__class__.__name__}"
            )

    def format_for_evalai(self, report):
        return generate_prediction(report)

    def align_evalai_report_order(self, report):
        return align_evalai_report_order(report, self.annotation_db)

    def format_for_overwrite_obs_labels(self, report):
        # overwrite obs and labels
        return generate_new_obs_labels(report)

    def __len__(self):
        return len(self.annotation_db)


def generate_prediction(report):
    scores = torch.nn.functional.softmax(report.logits, dim=1)
    _, labels = torch.max(scores, 1)
    # Probability that the meme is hateful, (1)
    probabilities = scores[:, 1]

    predictions = []

    for idx, image_id in enumerate(report.id):
        proba = probabilities[idx].item()
        label = labels[idx].item()
        predictions.append({ID_STR: image_id.item(), "proba": proba, "label": label})
    return predictions


def align_evalai_report_order(report, annotation):
    # get annotation index,
    # hateful memes requires to have the same order
    id_seq = [s.get(ID_STR) for s in annotation]

    report_set = {}
    for rpt in report:
        report_set.update({rpt.get(ID_STR): rpt})

    new_report = []
    for id in id_seq:
        itm = report_set.get(id)
        assert (
            itm is not None
        ), f"{id} is missing in output from the report, but is in annotation"
        new_report.append(itm)

    return new_report


def generate_new_obs_labels(report):
    predictions = []

    for idx, image_id in enumerate(report.id):
        label = report.targets[idx].item()
        org_text = report.org_tokens[idx]
        adv_text = report.adv_tokens[idx]

        org_text = " ".join([w for w in org_text if len(w.strip()) > 0])
        adv_text = " ".join([w for w in adv_text if len(w.strip()) > 0])

        img_id = image_id.item()
        item = {
            ID_STR: img_id,
            "img": "{}.png".format(img_id),
            "label": label,
            "text": adv_text,
            "org_text": org_text,
            "org_image": report.clean_image[idx],
            "image": report.noisy_image[idx],
        }
        if hasattr(report, "image_delta"):
            item.update({"image_delta": report.image_delta[idx]})
        predictions.append(item)
    return predictions
