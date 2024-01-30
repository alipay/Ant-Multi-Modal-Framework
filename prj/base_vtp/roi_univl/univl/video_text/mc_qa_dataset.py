# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import collections
import random

import torch

from antmmf.structures import NestedTensor, SampleList
from antmmf.structures import Sample
from antmmf.datasets.database.annotated import AnnotatedDatabase
from .ret_dataset import MMFUnivlVideoDataset


class RetrivalAnnotated_MC_QA(AnnotatedDatabase):
    def process_annotation(self, annotation_database):
        # group data by clip_name
        return_db = collections.defaultdict(list)
        for video_idx, item in enumerate(annotation_database):
            # msrvtt_mc_qa test format:
            # {{'qid': 'xxx', 'clip_name': 'xxx', 'title': 'xxx', 'answer': '2', 'options': [xxx...]}}
            new_item = {
                "clip_name": item["clip_name"],
                "qid": item["qid"],
                "video_idx": video_idx,
                "label": item["answer"],
                "options": item["options"],
                "type": item.get("type", "video"),
            }
            return_db[video_idx].append(new_item)

        return return_db

    def preprocess_item(self, item):
        # 随机选择一个 video_caption_list,
        # mil-nce loss假设一个batch中不存在两个相同的video
        return random.choice(item)


class AddCaptionMixin(object):
    def add_caption_for_mc_qa(self, sample_info, sample):
        assert hasattr(self, "caption_processor")
        if "options" in sample_info:
            # add caption for input
            cap_info_dict = {}
            include_key_list = [
                "input_ids",
                "input_mask",
                "segment_ids",
                "lm_label_ids",
                "tokens",
                "source_len",
                "text",
                "cls_id",
                "sep_id",
            ]
            tensor_key_list = ["input_ids", "input_mask", "segment_ids", "lm_label_ids"]
            for key in include_key_list:
                cap_info_dict["caption_" + key] = []

            for cap_info in sample_info["options"]:
                cap_input = self.caption_processor({"text": cap_info})
                for key, val in cap_input.items():
                    cap_info_dict["caption_" + key].append(val)
            # tensor stack
            for key in tensor_key_list:
                cap_info_dict["caption_" + key] = torch.stack(
                    cap_info_dict["caption_" + key]
                )
            sample["caption_options"] = [cap_info_dict]
            sample["caption_length"] = len(sample_info["options"])


class MMFUnivlVideoDataset_MC_QA(MMFUnivlVideoDataset, AddCaptionMixin):

    NAME = "video_multi_choice_qa"

    def _build_annotation_db(self):
        return super(MMFUnivlVideoDataset, self)._build_annotation_db(
            database_cls=RetrivalAnnotated_MC_QA
        )

    def _get_one_item(self, sample_info, video_info):
        sample_info.update(video_info)
        current_sample = Sample()

        # step1： add video
        self.add_video(
            sample_info,
            current_sample,
            add_default=self.config.get("allow_video_miss", False),
        )

        if current_sample.image_data is None:
            return None

        # step2: caption for multi-choice QA
        self.add_caption_for_mc_qa(sample_info, current_sample)

        # step3: add cluster id
        self.add_label(sample_info, current_sample)

        return Sample(current_sample)

    def collate_fn(self, batch):
        # filter None sample
        batch = [x for x in batch if x is not None]
        bsz = len(batch)
        if bsz == 0:
            return SampleList()
        n_clips, num_frames = batch[0].image_data.shape[:2]
        # flatten batch: each sample: bsz, n_clips, num_frames, c, h, w -> bsz*n_clips*num_frames,c,h,w
        tensor_list = []  # list of [cxhxw], length: bsz*n_clips*num_frames
        for sample in batch:
            img_shape = sample.image_data.shape
            img_tensor = sample.image_data.view(n_clips * num_frames, *img_shape[2:])
            tensor_list.extend(img_tensor.unbind(0))
        pad_imgs, pad_masks = NestedTensor.from_tensor_list(tensor_list).decompose()
        # reshape back
        pad_imgs = pad_imgs.view(bsz, n_clips * num_frames, *pad_imgs.shape[-3:])
        pad_masks = pad_masks.view(bsz, n_clips * num_frames, *pad_masks.shape[-2:])
        for i in range(len(batch)):
            batch[i].image_data = [pad_imgs[i]]
            batch[i].image_pad_mask = [pad_masks[i]]
            batch[i].image_n_clips = [n_clips]
            batch[i].image_num_frames = [num_frames]
        samplelist = SampleList(batch)
        return samplelist
