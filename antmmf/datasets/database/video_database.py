# Copyright (c) 2023 Ant Group and its affiliates.
import collections
import csv
import io
import os
import warnings
from dataclasses import dataclass
from typing import Any, Union, Dict, List, AnyStr

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from numpy.random import randint

from antmmf.common import Configuration, AntMMFConfig
from antmmf.datasets.database.image_database import ImageDatabase
from antmmf.utils.general import get_absolute_path


class OnlineVideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data["url"]

    @property
    def num_frames(self):
        return self._data.get("num_frame", 1)

    @property
    def start_frame(self):
        return self._data.get("start_frame", 1)

    @property
    def video_text(self):
        return self._data.get("video_text", "")

    @property
    def label(self):
        return self._data["label"]

    @property
    def image_seq(self):
        return self._data["image_seq"]

    @property
    def image_id(self):
        return self._data["image_id"]

    def update(self, new_content):
        self._data.update(new_content)
        return self._data


class KeyFramesDatabase(ImageDatabase):
    """
    Video is represented as key frames.
    Dataset that is originally designed for security,
    General format that we have standardize follows:
    {
        data: [
            {
                'id': DATASET_SET_ID,
                'label': <directory>,
                'image_seq': [<image>]
                <image> is a (H, W, C) numpy ndarray in int8
            }
        ]
    }
    """

    def __init__(
        self,
        csv_file,
        num_segments=3,
        new_length=1,
        random_shift=True,
        test_mode=False,
        remove_missing=False,
        dense_sample=False,
        twice_sample=False,
        record_type="online_video",
    ):
        self.num_segments = num_segments
        self.new_length = new_length
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        self.record_type = record_type

        self._load_imdb(csv_file)

        if self.dense_sample:
            print("=> Using dense sample for the dataset...")
        if self.twice_sample:
            print("=> Using twice sample for the dataset...")

    def _load_imdb(self, csv_file):

        tmp = list()

        with open(csv_file, encoding="utf-8") as csv_fd:
            csv_reader = csv.reader(csv_fd)
            for idx, (
                md5_id,
                scene,
                content,
                url,
                picture_ids,
                label_l2,
                gmt_create,
                gmt_modified,
                dt,
            ) in enumerate(csv_reader):
                if idx == 0:
                    continue
                if self.record_type == "online_video":
                    tmp.append(
                        {
                            "url": url,
                            "label": label_l2,
                            "image_id": md5_id,
                            "scene": scene,
                            "start_frame": 1,
                        }
                    )

            if self.record_type == "online_video":
                self.data = [OnlineVideoRecord(item) for item in tmp]
            else:
                raise Exception("not support record type {}".format(self.record_type))

        self.start_idx = 0

    @staticmethod
    def normal_sample(num_frames, num_segments, clip_length, stage):
        if stage == "train":
            average_duration = (num_frames - clip_length + 1) // num_segments
            if average_duration > 0:
                offsets = np.arange(num_segments) * average_duration + randint(
                    average_duration, size=num_segments
                )
            elif num_frames > num_segments and num_frames > clip_length:
                # there exists overlap among clips, each clip contains one frame at least
                offsets = np.sort(
                    randint(num_frames - clip_length + 1, size=num_segments)
                )
            else:
                offsets = np.zeros((num_segments,), dtype=np.int32)
        else:
            if num_frames > num_segments + clip_length - 1:
                average_duration = (num_frames - clip_length + 1) / float(num_segments)
                offsets = np.int32(
                    [
                        int(average_duration / 2.0 + average_duration * x)
                        for x in range(num_segments)
                    ]
                )
            else:
                offsets = np.zeros((num_segments,), dtype=np.int32)

        return offsets + 1

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [
                (idx * t_stride + start_idx) % record.num_frames
                for idx in range(self.num_segments)
            ]
            return np.array(offsets) + 1
        else:  # normal sample
            return KeyFramesDatabase.normal_sample(
                record.num_frames, self.num_segments, self.new_length, "train"
            )

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [
                (idx * t_stride + start_idx) % record.num_frames
                for idx in range(self.num_segments)
            ]
            return np.array(offsets) + 1
        else:
            return KeyFramesDatabase.normal_sample(
                record.num_frames, self.num_segments, self.new_length, "val"
            )

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list:
                offsets += [
                    (idx * t_stride + start_idx) % record.num_frames
                    for idx in range(self.num_segments)
                ]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
                + [int(tick * x) for x in range(self.num_segments)]
            )

            return offsets + 1
        else:
            return KeyFramesDatabase.normal_sample(
                record.num_frames, self.num_segments, self.new_length, "val"
            )

    def _load_video_url(self, url, idx):
        video = cv2.VideoCapture(url)
        frame = None
        while video.isOpened():
            video.set(1, idx)
            ret, frame = video.read()
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                break
            except Exception:
                break

        video.release()

        return [frame]

    def __getitem__(self, index):
        data = self.data[index + self.start_idx]

        info = self.get(data, [0])
        ret = {}
        ret["video_capture"] = info.image_seq
        ret["targets"] = info.label
        ret["image_id"] = data.image_id
        ret["tokens"] = ""
        ret["url"] = info.path

        if len(info.image_seq) == 0:
            return None

        return ret

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                if self.record_type in "online_video":
                    seg_imgs = self._load_video_url(
                        record.path, record.start_frame + p - 1
                    )
                else:
                    raise NotImplementedError(
                        f"not supported format to get {self.record_type}"
                    )

                if seg_imgs is not None:
                    images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
        import copy

        ret = copy.deepcopy(record)
        ret.update({"image_seq": images})
        return ret


class MultiSourceLoader(object):
    """
    Loading videos from multiple sources, include:
    1. lmdb database
    2. video dir where .mp4, .avi format videos placed
    3. video keyframes format
    """

    def __init__(self, base_paths: List[AnyStr]):
        self.lmdb_txns = []
        self.video_dirs, self.lmdb_txns = [], []
        for path in base_paths:
            if path.endswith(".lmdb"):  # 读取lmdb格式video目录
                import lmdb

                # https://codeslake.github.io/research/pytorch/How-to-use-LMDB-with-PyTorch-DataLoader/
                env = lmdb.open(
                    path, readonly=True, create=False, lock=False
                )  # readahead=not _check_distributed()
                txn = env.begin(buffers=True)
                self.lmdb_txns.append(txn)
            elif os.path.isdir(path):  # 读取video_dir
                self.video_dirs.append(path)

    def get_keyframes_dir(self, video_name):
        key_frame_dir = None
        for vd in self.video_dirs:
            video_path = os.path.join(vd, video_name)
            if os.path.isdir(video_path):
                key_frame_dir = video_path
                break
        return key_frame_dir

    def read_bytes(self, video_name):
        video_bytes = None
        video_id = os.path.splitext(os.path.split(video_name)[1])[0]
        for txn in self.lmdb_txns:
            video_bytes = txn.get(str(video_id).encode("utf-8"))
            if video_bytes is not None:
                break
        if video_bytes is None:
            for vd in self.video_dirs:
                video_path = os.path.join(vd, video_name)
                if os.path.isfile(video_path):
                    video_bytes = open(video_path, "rb").read()
                    break
        return video_bytes


class VideoClipsDatabase(torch.utils.data.Dataset):
    """
    Dataset for input video format, such as mp4, avi .etc
    Loading video by sparse sampling strategy:
    see detail:
    Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling
    https://arxiv.org/abs/2102.06183
    """

    @dataclass
    class Config(AntMMFConfig):
        train_ensemble_n_clips: int = 8  #
        test_ensembel_n_clips: int = 16
        num_frm: int = 2  # sample frames of each clip

    def __init__(
        self,
        path: str,
        annotation_db: Any,
        config: Union[Config, Configuration, Dict] = None,
        dataset_type="train",
        transform=None,
        is_valid_file=None,
        video_field_key=None,
        **kwargs,
    ):
        """Initialize an instance of VideoDatabase, handling

        Args:
            path (str): Path to images folder
            annotation_db (AnnotationDB, optional): Annotation DB to be used
                to be figure out image paths. Defaults to None.
            transform (callable, optional): Transform to be called upon loaded image.
                Defaults to None.
            loader (callable, optional): Custom loader for image which given a path
                returns a PIL Image. Defaults to torchvision's default loader.
            is_valid_file (callable, optional): Custom callable to filter out invalid
                files. If image is invalid, {"images": []} will returned which you can
                filter out in your dataset. Defaults to None.
            video_field_key(list, optional): indicate image keys of annotation_db to load.
                If image_field_keys not indicated, using `_get_attrs` method to get possible
                images to load.
        """
        super().__init__()
        self.base_path = get_absolute_path(path)
        if not isinstance(self.base_path, list):
            self.base_path = [self.base_path]
        assert annotation_db is not None, "Annotation Database Not Provided"
        self._transform = transform
        self._annotation_db = annotation_db
        self.is_valid_file = is_valid_file
        self.video_field_key = video_field_key
        self.dataset_type = dataset_type

        # sparse sampling parameters
        # https://github.com/jayleicn/ClipBERT/blob/main/src/tasks/run_video_retrieval.py#L87-L89
        self.config = self.__class__.Config.create_from(config, **kwargs)
        self.ensemble_n_clips = (
            self.config.train_ensemble_n_clips
            if self.dataset_type == "train"
            else self.config.test_ensembel_n_clips
        )
        self.num_frm = self.config.num_frm

        # support lmdb database for faster reading, convert video to lmdb:
        # prj/univl/scripts/lmdb_conversion.py
        self.binary_reader = MultiSourceLoader(self.base_path)

        from antmmf.utils.video_utils import VideoReader

        self.video_reader = VideoReader(
            training=self.dataset_type == "train", num_frm=self.num_frm
        )

    @property
    def annotation_db(self):
        return self._annotation_db

    @annotation_db.setter
    def annotation_db(self, annotation_db):
        self._annotation_db = annotation_db

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        if isinstance(transform, collections.abc.MutableSequence):
            transform = torchvision.transforms.Compose(transform)
        self._transform = transform

    def get_video_bytes(self, video_path):
        return self.binary_reader.read_bytes(video_path)

    def __len__(self):
        return len(self.annotation_db)

    def __getitem__(self, idx):
        item = self.annotation_db[idx]
        return self.get(item)

    def _get_attrs(self, item):
        """Returns possible attribute that can point to image id

        Args:
            item (Object): Object from the DB

        Returns:
            List[str]: List of possible images that will be copied later
        """
        video = None
        pick = None
        attrs = self._get_possible_attrs()

        for attr in attrs:
            video = item.get(attr, None)
            if video is not None:
                pick = attr
                break

        # Handle public dataset format
        if pick == "clip_name" and len(video.split(".")) == 1:
            return video + ".mp4"  # msrvtt
        else:
            return video

    def _get_possible_attrs(self):
        return ["video", "clip_name", "video_path", "video_name", "video_id"]

    def get(self, item):
        if self.video_field_key is None:
            video_name = self._get_attrs(item)
        else:
            video_name = item.get(self.video_field_key)

        return_info = {
            "video": None,
            "n_clips": self.ensemble_n_clips,
            "num_frm": self.num_frm,
            "video_mask": None,
        }

        # filter videos you do not want to load
        if self.is_valid_file is not None and callable(self.is_valid_file):
            if not self.is_valid_file(video_name):
                return return_info

        # load video
        key_frames_dir = self.binary_reader.get_keyframes_dir(video_name)
        video = None
        if key_frames_dir is not None:
            video, frame_idxs, _ = self.video_reader.read_frames_from_img_dir(
                key_frames_dir,
                self.ensemble_n_clips,
                fix_start=None,
                frame_resample="uniform",
            )
        else:
            video_bytes = self.get_video_bytes(video_name)
            if video_bytes is not None:
                try:
                    video, frame_idxs, _ = self.video_reader.read_frames_decord(
                        io.BytesIO(video_bytes),
                        self.ensemble_n_clips,
                        begin_time=item.get("start", None),
                        end_time=item.get("end", None),
                    )
                except Exception:
                    warnings.warn(f"loading video failed:{video_name}")

        # FIXME: use mask to avoid insufficient frames
        if video is not None and video.shape[0] != self.ensemble_n_clips * self.num_frm:
            warnings.warn(
                f"loading video shape failed:{video_name} :"
                f"{video.shape[0]}!={self.ensemble_n_clips * self.num_frm}"
            )
            video = None
        video_mask = None

        if self.transform is not None and video is not None:
            video = self.transform(video)  # TCHW, normalized by mean/std
            # import matplotlib.pylab as plt
            # frames = video.permute(0, 2, 3, 1)
            # for i in range(1, frames.shape[0] + 1):
            #     plt.subplot(1, frames.shape[0], i)
            #     plt.imshow(((frames[i - 1]*0.225 + 0.485)*255).to(torch.uint8))
            # plt.show()

        return_info.update({"video": video, "video_mask": video_mask})
        return return_info
