# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import collections
import copy
import os.path as osp
import random

import torch

from antmmf.common import Configuration
from antmmf.common.registry import registry
from antmmf.datasets.base_dataset import BaseDataset
from antmmf.datasets.database.annotated import AnnotatedDatabase
from antmmf.structures import NestedTensor, SampleList
from antmmf.structures import Sample
from antmmf.utils.general import get_antmmf_root, check_required_keys
from ...roi.dataset import AddCaptionMixin


class RetrivalAnnotated(AnnotatedDatabase):
    def process_annotation(self, annotation_database):
        self.vid_map = {}
        self.tid_map = {}
        self.t2v = collections.defaultdict(list)
        self.v2t = collections.defaultdict(list)
        # group data by clip_name
        processed_db = collections.defaultdict(list)
        for item in annotation_database:
            video = item["clip_name"]
            processed_db[video].append(item)
            if video not in self.vid_map:
                vid = len(self.vid_map)
                self.vid_map[video] = vid
            else:
                vid = self.vid_map[video]

            cap = item["caption"]
            if cap not in self.tid_map:
                tid = len(self.tid_map)
                self.tid_map[cap] = tid
            else:
                tid = self.tid_map[cap]

            self.t2v[tid].append(vid)
            self.v2t[vid].append(tid)

        return_db = collections.defaultdict(list)
        global_idx = 0
        for video_idx, (clip_name, item_list) in enumerate(processed_db.items()):
            # msrvtt train format: {"caption": "xxx", "clip_name": "video9771", "sen_id": 0}
            # msrvtt test format: {"caption": "xxx", "clip_name": "video9771", "retrieval_key": "ret1"}
            sorted_item = sorted(item_list, key=lambda x: x.get("sen_id", 0))
            captions = [x["caption"] for x in sorted_item]
            dtypes = [x.get("type", "video") for x in sorted_item]
            # unique identifier for each retrival item
            caption_keys = [
                x.get("retrieval_key") or x.get("sen_id") for x in sorted_item
            ]
            for cap, key, dtype in zip(captions, caption_keys, dtypes):
                vid, tid = self.vid_map[clip_name], self.tid_map[cap]
                tid_list, vid_list = self.v2t[vid], self.t2v[tid]
                new_item = {
                    "clip_name": clip_name,
                    "vid": vid,
                    "caption": cap,
                    "tid": tid,
                    "v2tid_list": tid_list,
                    "t2vid_list": vid_list,
                    "key": key,
                    "type": dtype,
                }
                index_key = (
                    video_idx
                    if self.kwargs.get("dataset_type") == "train"
                    else global_idx
                )
                return_db[index_key].append(new_item)
                global_idx += 1
        return return_db

    def preprocess_item(self, item):
        # 随机选择一个 video_caption_list,
        # mil-nce loss假设一个batch中不存在两个相同的video
        return random.choice(item)


class AddVideoMixin(object):
    def add_video(self, sample_info, sample, add_default=True):
        assert check_required_keys(sample_info, ["video", "n_clips", "num_frm"])

        if sample_info["video"] is None:
            if add_default:
                sample.image_data = torch.zeros(
                    (sample_info["n_clips"], sample_info["num_frm"], 3, 50, 50)
                )
            else:
                sample.image_data = None
        else:
            # num_clips, num_frames, 3, h, w
            sample.image_data = sample_info["video"].view(
                sample_info["n_clips"],
                sample_info["num_frm"],
                *sample_info["video"].shape[1:]
            )

        if "vid" in sample_info:
            sample.image_vid = torch.tensor(sample_info["vid"], dtype=torch.long)
        if "v2tid_list" in sample_info:
            sample.image_tid_list = sample_info["v2tid_list"]

        sample.video_mask = sample_info["video_mask"]
        if "clip_name" in sample_info:
            sample.clip_name = sample_info["clip_name"]
        else:
            sample.clip_name = sample_info["video_name"]


class AddClusterIdMixin(object):
    def add_label(self, sample_info, sample):
        if sample_info.get("label") is not None:
            sample["targets"] = torch.tensor(
                int(sample_info["label"]), dtype=torch.long
            )


class MMFUnivlVideoDataset(
    BaseDataset, AddCaptionMixin, AddVideoMixin, AddClusterIdMixin
):

    NAME = "video_text_retrieval"

    def __init__(self, dataset_type, config):
        super().__init__(self.__class__.NAME, dataset_type, config)
        self.first_call = True

    def setup_extras(self, dataset_type, config, *args, **kwargs):
        self.video_db.annotation_db = self.annotation_db
        # for easy batching
        if self.dataset_type == "train":
            self.video_db.transform = self.train_frame_processor
        else:
            self.video_db.transform = self.test_frame_processor

    def __len__(self):
        return len(self.video_db)

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

        # step2: caption
        self.add_caption(sample_info, current_sample)

        # step3: add cluster id
        self.add_label(sample_info, current_sample)

        return Sample(current_sample)

    def get_item(self, idx):
        sample_info = copy.deepcopy(self.annotation_db[idx])
        video_info = self.video_db[idx]
        return self._get_one_item(sample_info, video_info)

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
            batch[i].image_data = pad_imgs[i]
            batch[i].image_pad_mask = pad_masks[i]
            batch[i].image_n_clips = n_clips
            batch[i].image_num_frames = num_frames
        samplelist = SampleList(batch)
        return samplelist

    def get_annotations(self, annotation_type):
        if self.first_call:
            self.annotation_data = {"labels": []}
            for x in self.annotation_db.data:
                self.annotation_data["labels"].append(x["label"])
            self.first_call = False
        assert annotation_type in ["labels"]
        return self.annotation_data[annotation_type]

    def _build_annotation_db(self):
        return super()._build_annotation_db(
            database_cls=RetrivalAnnotated, dataset_type=self._dataset_type
        )

    def _build_video_db(self):
        return super()._build_video_db(**self.config, dataset_type=self._dataset_type)

    # disable parent class download method
    def _download_requirement(
        self, config, requirement_key, requirement_variation="defaults"
    ):
        return None


if __name__ == "__main__":
    test_yaml = "configs/univl/video/finetune_retrieval/univl_video_text_retrieval.yml"
    config_yaml_file = osp.join(get_antmmf_root(), "..", test_yaml)
    config = Configuration(config_yaml_file)
    config.freeze()
    dataset_config = (
        config.task_attributes.univl_task.dataset_attributes.video_text_retrieval
    )
    from antmmf.utils.logger import Logger

    registry.register("writer", Logger(config))

    train_dataset = MMFUnivlVideoDataset("train", dataset_config)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=train_dataset.collate_fn,
    )

    for i_batch, batched in enumerate(train_loader):
        print(batched)
