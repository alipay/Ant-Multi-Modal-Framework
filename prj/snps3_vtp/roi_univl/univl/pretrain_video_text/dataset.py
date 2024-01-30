# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import codecs
import collections
import copy
import json
import math
import os
import os.path as osp
import warnings

import numpy as np
import torch
from PIL import Image, ImageFile

from torchvision import transforms
from dataclasses import dataclass

from antmmf.common import AntMMFConfig
from antmmf.common.registry import registry
from antmmf.structures import SampleList, NestedTensor
from antmmf.datasets.base_dataset import _get_config_path
from antmmf.datasets.concat_dataset import AntMMFConcatDataset
from antmmf.datasets.database.annotated import AnnotatedDatabase
from antmmf.datasets.database.video_database import VideoClipsDatabase
from antmmf.utils.distributed_utils import get_rank, get_world_size
from antmmf.utils.general import get_absolute_path
from antmmf.utils.general import get_antmmf_root
import random
from antmmf.common import Configuration

from ..video_text.ret_dataset import MMFUnivlVideoDataset


ImageFile.LOAD_TRUNCATED_IMAGES = True  # PIL.UnidentifiedImageError
Image.MAX_IMAGE_PIXELS = None  # PIL.Image.DecompressionBombError


class PretrainAnnotated(AnnotatedDatabase):
    def process_annotation(self, annotation_database):
        # group data by clip_name
        processed_db = collections.defaultdict(list)
        for item in annotation_database:
            processed_db[item["clip_name"]].append(item)

        return_db = collections.defaultdict(list)
        vid_map = dict()
        for video_idx, (clip_name, item_list) in enumerate(processed_db.items()):
            # univl pretrain data format:
            # image format:
            # {"caption": "xxx", "clip_name": "video9771", "type": "image", "start":None, 'end': None}
            # video format:
            # {"caption": None, "cilp_name": "video9771", "type": "video", "start": None, "end": None}
            sorted_item = sorted(item_list, key=lambda x: x.get("sen_id", 0))

            # filter annotations
            captions = [x["caption"] for x in sorted_item]
            dtypes = [x.get("type", "video") for x in sorted_item]
            starts = [x.get("start", None) for x in sorted_item]
            ends = [x.get("end", None) for x in sorted_item]
            caption_keys = [
                x.get("retrieval_key") or x.get("sen_id") for x in sorted_item
            ]

            for cap, key, dtype, st, et in zip(
                captions, caption_keys, dtypes, starts, ends
            ):
                new_item = {
                    "clip_name": clip_name,
                    "caption": cap,
                    "video_idx": video_idx,
                    "key": key,
                    "type": dtype,
                    "start": st,
                    "end": et,
                }
                # incase a video has multiple texts in pretraining data
                vid = f"{video_idx}_{st}_{et}"
                if vid not in vid_map:
                    global_idx = len(vid_map)
                    vid_map[vid] = global_idx
                vid_idx = vid_map[vid]
                return_db[vid_idx].append(new_item)
        return return_db

    def preprocess_item(self, item):
        # 随机选择一个 video_caption_list,
        # mil-nce loss假设一个batch中不存在两个相同的video
        return random.choice(item)


class AutoSplitAnnotated(PretrainAnnotated):
    """
    Avoid loading whole annotation file and separate large training data annotations into NUM_GPUS splits.
    Each gpu's process will only loads partly annotation belonging to its own, such will save memory by
    NUM_GPUS times.

    AntMMF will use 'distributed_sampler' for DDP training, with each gpu loading its corresponding splits.
    For AutoSplitAnnotated, the annotation is already split for each gpu, so there is no need to further split
    again.
    **Note**: To use AutoSplitAnnotated in DDP training, users must set AntMMF training flag:
         training_parameters.distributed_sampler.type random_sampler
    """

    def __init__(self, annotation_path, anno_type="normal", **kwargs):
        self.annotation_path = annotation_path
        self.start = 0
        self.end = self._get_length()
        self._init_iter_range()
        self._get_sub_data()

    def _get_length(self):
        num_annotations = 0
        for line in codecs.open(self.annotation_path, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            num_annotations += 1
        return num_annotations

    def _get_sub_data(self):
        num_annotations = 0
        self.sub_data = []
        for line in codecs.open(self.annotation_path, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            if self.iter_start <= num_annotations < self.iter_end:
                self.sub_data.append(json.loads(line))
            num_annotations += 1
        self.sub_data = self.process_annotation(self.sub_data)
        # assert len(self.sub_data) == self.per_worker

    def __len__(self):  # 所有gpu dataset长度一致
        return self.per_worker

    def __getitem__(self, idx):
        if not 0 <= idx < len(self.sub_data):
            # writer = registry.get("writer")
            # worker_id = get_rank()
            # writer.write(
            #     f"[Error index]: worker_id:{worker_id} iter_start:{self.iter_start} "
            #     f"iter_end:{self.iter_end} idx:{idx} sub_data_len:{len(self.sub_data)}",
            #     "info",
            #     log_all=True,
            # )
            idx = random.randint(0, len(self.sub_data) - 1)
        return self.preprocess_item(self.sub_data[idx])

    def _init_iter_range(self):
        writer = registry.get("writer")
        worker_id = get_rank()
        # 每个gpu读取不同的annotation
        # split workload
        self.per_worker = int(
            math.ceil((self.end - self.start) / float(get_world_size()))
        )

        # [self.iter_start, self.iter_end)
        self.iter_start = self.start + worker_id * self.per_worker
        self.iter_end = min(self.iter_start + self.per_worker, self.end)
        # padding for ensuring each process has the same data length
        if self.iter_end - self.iter_start < self.per_worker:
            self.iter_start = self.iter_end - self.per_worker

        writer.write(
            f"worker_id:{worker_id} iter_start:{self.iter_start} iter_end:{self.iter_end}",
            "info",
            log_all=True,
        )


class ImageVideoDatabase(VideoClipsDatabase):
    def __init__(self, image_path, video_path, annotation_db, **kwargs):
        super().__init__(video_path, annotation_db, **kwargs)
        self.image_base_path = get_absolute_path(image_path) if image_path else None
        self.image_transform = transforms.PILToTensor()

    def _load_inflated_image(self, image_path):
        try:  # do not assume images are valid
            image_raw = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            warnings.warn(f"image_path not found:{image_path}")
            return None
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


class ImageClipMixDatabase(ImageVideoDatabase):
    @dataclass
    class ASRSamplerConfig(AntMMFConfig):
        with_long_context: bool = True
        train_n_pairs: int = 3
        test_n_pairs: int = 3
        min_words: int = 0
        min_time: float = 10.0

    def __init__(
        self,
        image_path,
        video_path,
        annotation_db,
        asr_path,
        asr_sampler_config=None,
        **kwargs,
    ):
        super().__init__(image_path, video_path, annotation_db, **kwargs)

        # params for loading video asr info
        assert asr_path is not None
        self.asr_dir = asr_path
        self.asr_sampler_config = ImageClipMixDatabase.ASRSamplerConfig.create_from(
            asr_sampler_config, **kwargs
        )
        if self.dataset_type == "train":
            self.n_pairs = self.asr_sampler_config.train_n_pairs
        else:
            self.n_pairs = self.asr_sampler_config.test_n_pairs
        self.min_time = self.asr_sampler_config.min_time
        self.min_words = self.asr_sampler_config.min_words
        self.with_long_context = self.asr_sampler_config.with_long_context

    def _load_video_asrs(self, video_name):
        # load video caption/asr/transcript info
        video_name = osp.basename(video_name)
        caption_file = osp.join(self.asr_dir, osp.splitext(video_name)[0])
        caption_list = []
        if not osp.exists(caption_file):
            return caption_list
        try:
            with open(caption_file, "r", encoding="utf-8") as f:
                caption_dict = json.load(f)
                caption_list = self._get_text(caption_dict, self.n_pairs)
        except BaseException:
            warnings.warn(f"caption file not found:{caption_file}")
        return caption_list

    def _get_text(self, data_dict, n_pair_max):
        # random sample one new long clip from current video
        # Reference:
        # https://github.com/microsoft/UniVL/blob/0a7c07f566a3b220731f4abcaa6e1ee59a686596/dataloaders/dataloader_howto100m.py#L230
        n_caption = len(data_dict["start"])
        if n_pair_max == -1:
            k = n_caption
            r_ind = range(n_caption)
        else:
            k = n_pair_max
            if k <= n_caption:
                r_ind = np.random.choice(range(n_caption), k, replace=False)
            else:
                r_ind_must = np.array(range(n_caption))
                r_ind_rand = np.random.choice(
                    range(n_caption), k - n_caption, replace=True
                )
                r_ind = np.concatenate((r_ind_must, r_ind_rand), axis=0)
            np.random.shuffle(r_ind)

        caption_list = []
        texts = set()
        for i in range(k):
            ind = r_ind[i]
            text, start, end = self._get_single_transcript(
                data_dict, ind, with_long_context=self.with_long_context
            )
            # avoid adding duplicated video-text pairs
            # FIXME: this will cause various batch_size across GPU devices
            # FIXME: limit text distance
            if text not in texts:
                texts.add(text)
                caption_list.append({"caption": text, "start": start, "end": end})
        return caption_list

    def _get_single_transcript(self, data_dict, ind, with_long_context=True):
        start, end = ind, ind
        words = str(data_dict["text"][ind])
        diff = data_dict["end"][end] - data_dict["start"][start]
        while with_long_context and (
            len(words) < self.min_words or diff < self.min_time
        ):
            if start > 0 and end < len(data_dict["end"]) - 1:
                next_words = str(data_dict["text"][end + 1])
                prev_words = str(data_dict["text"][start - 1])
                d1 = data_dict["end"][end + 1] - data_dict["start"][start]
                d2 = data_dict["end"][end] - data_dict["start"][start - 1]
                if (self.min_time > 0 and d2 <= d1) or (
                    self.min_time == 0 and len(next_words) <= len(prev_words)
                ):
                    start -= 1
                    words = prev_words + words
                else:
                    end += 1
                    words = words + next_words
            elif start > 0:
                words = str(data_dict["text"][start - 1]) + words
                start -= 1
            elif end < len(data_dict["end"]) - 1:
                words = words + str(data_dict["text"][end + 1])
                end += 1
            else:
                break
            diff = data_dict["end"][end] - data_dict["start"][start]
        return words, data_dict["start"][start], data_dict["end"][end]

    def __getitem__(self, idx):
        item = self.annotation_db[idx]
        dtype = item["type"]
        load_asr = (
            False
            if item.get("caption") is not None and len(item["caption"].strip()) > 0
            else True
        )
        # random sample one clip from video
        if dtype == "video" and load_asr:
            # default
            item_info = {
                "video": None,  # frame format:rgb, (n_clips*num_frm, C, H, W)
                "n_clips": self.ensemble_n_clips,
                "num_frm": self.num_frm,
                "video_mask": None,
                "caption": "",
                "start": None,
                "end": None,
            }
            vid = item["clip_name"]
            caption_list = self._load_video_asrs(vid)
            for one_cap in caption_list:
                # extract corresponding video for asr
                new_item = copy.deepcopy(item)
                new_item.update(one_cap)
                sub_item_info = self.get(new_item)
                if sub_item_info["video"] is not None:
                    # valid asr-clip pairs found
                    item_info.update(sub_item_info)
                    item_info.update(one_cap)
                    break
        else:
            item_info = self.get(item)
        return item_info


class MMFVideoPretrainDataset(MMFUnivlVideoDataset):

    NAME = "video_text_pretrain"

    def _build_annotation_db(self):
        annotations = self.config.get("annotations", {}).get(self._dataset_type, [])
        # User can pass a single string as well
        if isinstance(annotations, str):
            annotations = [annotations]
        assert len(annotations) > 0
        datasets = []
        for imdb_idx in range(len(annotations)):
            annotation_path = self._get_absolute_path(annotations[imdb_idx])
            # For large training annotations:
            # use AutoSplitAnnotated need to set
            # training_parameters.distributed_sampler.type random_sampler
            if (
                self._dataset_type == "train"
                and self.config.get("annotations_loader") == "auto_split"
            ):
                dataset = AutoSplitAnnotated(annotation_path)
            else:
                # antmmf validation sampler will always be sequential sampler
                dataset = PretrainAnnotated(annotation_path)
            datasets.append(dataset)
        return AntMMFConcatDataset(datasets)

    def _build_video_db(self):
        video_path, image_path, asr_path = ".", ".", None

        if self.config.get("use_videos", False):
            video_path = _get_config_path(
                self.config["videos"], self._dataset_type, self._index
            )
            video_path = self._get_absolute_path(video_path)

        if self.config.get("use_asrs", False):  # video asr format captions
            asr_path = _get_config_path(
                self.config["asrs"], self._dataset_type, self._index
            )
            asr_path = self._get_absolute_path(asr_path)

        if self.config.get("use_images", False):
            image_path = _get_config_path(
                self.config["images"], self._dataset_type, self._index
            )
            image_path = self._get_absolute_path(image_path)
        if asr_path is None:
            # no asr found, the holy image/video is related to one caption
            # no further process is needed
            database_cls = ImageVideoDatabase
        else:
            database_cls = ImageClipMixDatabase

        return database_cls(
            image_path,
            video_path,
            self.annotation_db,
            asr_path=asr_path,
            dataset_type=self._dataset_type,
            **self.config,
        )

    def collate_fn(self, batch):
        # filter None sample
        batch = [x for x in batch if x is not None]
        bsz = len(batch)
        sample_list = SampleList()
        if bsz == 0:
            return sample_list

        # 区分clip和n_pair维度的概念，clip需要做时序建模融合；n_pair是相对独立的样本
        n_clips, num_frames = batch[0].image_data.shape[:2]
        # flatten batch: each sample: bsz, n_clips, num_frames, c, h, w -> bsz*n_clips*num_frames,c,h,w
        tensor_list = []  # list of [cxhxw], length: bsz*n_clips*num_frames
        for sample in batch:
            img_shape = sample.image_data.shape
            n_clips, num_frames = img_shape[:2]
            img_tensor = sample.image_data.view(n_clips * num_frames, *img_shape[2:])
            tensor_list.extend(img_tensor.unbind(0))
        pad_imgs, pad_masks = NestedTensor.from_tensor_list(tensor_list).decompose()

        pad_imgs = pad_imgs.view(bsz, n_clips * num_frames, *pad_imgs.shape[-3:])
        pad_masks = pad_masks.view(bsz, n_clips * num_frames, *pad_masks.shape[-2:])
        batch_size = pad_imgs.size(0)

        # info
        sample_list.add_field("dataset_name", [batch[0].dataset_name] * batch_size)
        sample_list.add_field("dataset_type", [batch[0].dataset_type] * batch_size)

        # image
        sample_list.add_field("image_data", pad_imgs)
        sample_list.add_field("image_pad_mask", pad_masks)
        sample_list.add_field(
            "image_n_clips", torch.tensor([n_clips] * batch_size, dtype=torch.long)
        )
        sample_list.add_field(
            "image_num_frames",
            torch.tensor([num_frames] * batch_size, dtype=torch.long),
        )

        # caption
        sample_list.add_field(
            "caption_input_ids", torch.stack([s.caption_input_ids for s in batch], 0)
        )
        sample_list.add_field(
            "caption_input_mask", torch.stack([s.caption_input_mask for s in batch], 0)
        )
        sample_list.add_field(
            "caption_segment_ids",
            torch.stack([s.caption_segment_ids for s in batch], 0),
        )
        sample_list.add_field(
            "caption_lm_label_ids",
            torch.stack([s.caption_lm_label_ids for s in batch], 0),
        )
        # add raw captions
        sample_list.add_field(
            "caption_raw_input_ids",
            torch.stack([s.caption_raw_input_ids for s in batch], 0),
        )

        # false captions if have
        if self.config.get("add_false_caption", False):
            sample_list.add_field(
                "caption_false_input_ids",
                torch.stack([s.caption_false_input_ids for s in batch], 0),
            )
            sample_list.add_field(
                "caption_false_input_mask",
                torch.stack([s.caption_false_input_mask for s in batch], 0),
            )
        return sample_list


if __name__ == "__main__":
    test_yaml = "configs/univl/video/pretrain/base.yml"
    config_yaml_file = osp.join(get_antmmf_root(), "..", test_yaml)
    config = Configuration(config_yaml_file)
    config.freeze()
    dataset_config = (
        config.task_attributes.univl_task.dataset_attributes.video_text_pretrain
    )
    from antmmf.utils.logger import Logger

    registry.register("writer", Logger(config))

    train_dataset = MMFVideoPretrainDataset("train", dataset_config)

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
