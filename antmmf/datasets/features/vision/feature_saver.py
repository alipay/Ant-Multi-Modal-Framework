# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import tempfile
import traceback

import numpy as np
import torch
import tqdm
from PIL import Image

from antmmf.datasets.features.vision.base_extractor import OnlineFeatureExtractor
from antmmf.datasets.features.vision.detectron_feature import DetectronFeatureExtractor
from antmmf.datasets.features.vision.imagenet_feature import (
    ClassificationFeatureExtractor,
)
from antmmf.datasets.features.vision.video_feature import S3DGFeatureExtractor
from antmmf.utils.file_io import PathManager
from antmmf.utils.general import get_absolute_path

# workers for downloading batch images
DEFAULT_BATCH_DOWNLOAD_WORKERS = 10
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


class BaseSaver(object):
    def __init__(self, save_dir):
        PathManager.mkdir_if_not_exists(save_dir)
        self._save_dir = save_dir

    def get_save_dir(self):
        return self._save_dir

    def save_batch(self, image_paths, *feats_res):
        batch_size = len(image_paths)
        features, feature_infos = feats_res[0], [None] * batch_size
        if len(feats_res) == 2:
            feature_infos = feats_res[1]
            assert len(feature_infos) == len(features) == batch_size

        for img_path, feature, feature_info in zip(
            image_paths, features, feature_infos
        ):
            if isinstance(feature, torch.Tensor):
                feature = feature.cpu().numpy()
            self.save_one(img_path, feature, feature_info)

    def get_save_path(self, image_path):
        file_base_name = osp.basename(image_path)
        file_base_name = file_base_name.split(".")[0]
        info_file_base_name = file_base_name + "_info.npy"
        file_base_name = file_base_name + ".npy"

        save_path = osp.join(self._save_dir, file_base_name)
        info_file_base_name = osp.join(self._save_dir, info_file_base_name)
        should_save = not PathManager.exists(save_path)
        return should_save, save_path, info_file_base_name

    def save_one(self, image_path, feature, feature_info=None):
        should_save, save_path, info_file_base_name = self.get_save_path(image_path)

        if not should_save:
            return
        np.save(save_path, feature)
        if feature_info is not None:
            np.save(info_file_base_name, feature_info)


class DetectronFeatureSaver(BaseSaver):
    pass


class ClassificationFeatureSaver(BaseSaver):
    pass


class VideoFeatureSaver(BaseSaver):
    pass


class ImageCorpus(Dataset):
    def __init__(self, image_paths, saver: [BaseSaver] = None):
        self.image_paths = image_paths
        self.saver = saver

    def worker_func(self, img_path):
        pil_img = None
        try:
            pil_img = Image.open(img_path).convert("RGB")
            pil_img.verify()
        except Exception as ex:
            traceback.print_exc()
            pil_img = None
        finally:
            return pil_img

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        if self.saver is not None:
            if not self.saver.get_save_path(img_path)[0]:
                return None, None
        pil_img = self.worker_func(img_path)
        if pil_img is None:
            return None, None
        return img_path, np.asarray(pil_img)

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def collate_fn(sample_list: List[Tuple]):
        paths = [x[0] for x in sample_list if x[0] is not None]
        imgs = [x[1] for x in sample_list if x[1] is not None]
        if len(imgs) == 0:
            return [], []
        assert len(paths) == len(imgs)
        return paths, imgs


class FeatureManager(object):
    """
    Automatically manage extracted features saving, organize dirs as: {dataset_name}/{model_name}/{feature_name}/{split}
    it will create the hier like below:
        |____FasterRCNN-101-FPN
        | |____fc6
        | | |____train
        | | |____test
        |
        |____resnet152
        | |____block5_feature
        | | |____train
        | | |____test

    Usage:
    >>> from antmmf.datasets.features.vision.imagenet_feature import ResNetFeatureExtractor
    >>> fm = FeatureManager(feat_root_dir, dataset_name, split)
    >>> extractor = ResNetFeatureExtractor('resnet152')
    >>> fm.register_extractor(extractor)
    >>> fm.extract_and_save(image_path_list)
    """

    def __init__(self, feat_root_dir, dataset_name, split):
        assert split in ["train", "val", "test"]
        self._feat_root_dir = feat_root_dir
        self._dataset_name = dataset_name
        self._split = split
        self._extractors = None
        self._saver = None
        self._dirs = None

    def _get_hier_dirs(self, model_name, feature_name):
        """
        dataset/model_name/feature_layer/train|test
        """
        featdir = osp.join(self._feat_root_dir, model_name, feature_name, self._split)
        return featdir

    def _get_saver(self, feature_extractor):
        assert isinstance(feature_extractor, OnlineFeatureExtractor)
        featdir = self._get_hier_dirs(
            feature_extractor.get_model_name(), feature_extractor.get_feature_name()
        )
        factory = SaverFactory()
        saver = factory.create_saver(feature_extractor)
        return saver(featdir)

    def register_extractor(self, feature_extractor):
        self._extractor = feature_extractor
        self._saver = self._get_saver(feature_extractor)
        self._dir = self._saver.get_save_dir()

    def extract_and_save(self, image_paths, batch_size=16):
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]

        image_corpus = ImageCorpus(image_paths)
        image_loader = DataLoader(
            image_corpus,
            num_workers=DEFAULT_BATCH_DOWNLOAD_WORKERS,
            collate_fn=ImageCorpus.collate_fn,
            batch_size=batch_size,
            shuffle=False,
        )

        for batch_paths, batch_imgs in tqdm.tqdm(
            image_loader,
            total=int(len(image_corpus) / batch_size),
        ):
            # skip those images that had been extracted
            images = [Image.fromarray(np.uint8(img)) for img in batch_imgs]
            paths = [p for p in batch_paths]
            if len(images) == 0:
                continue
            feats_res = self._extractor.extract_features(images)
            self.save(paths, *feats_res)

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def save(self, image_paths, *feats_res):
        self._saver.save_batch(image_paths, *feats_res)

    def save_one(self, image_path, feature, feature_info=None):
        self._saver.save_one(image_path, feature, feature_info)


class VideoFeatureManager(FeatureManager):
    def extract_and_save(self, video_paths, batch_size=1):
        if not isinstance(video_paths, (list, tuple)):
            video_paths = [video_paths]
        for chunk in tqdm.tqdm(self._chunks(video_paths, 1)):
            feats_res = self._extractor.extract_features(
                chunk[0], time_batch_size=batch_size
            )
            self.save_one(chunk[0], feats_res)


# feature_extractor <-> saver
class SaverFactory(object):
    def create_saver(self, obj):
        if isinstance(obj, DetectronFeatureExtractor):
            return DetectronFeatureSaver
        elif isinstance(obj, ClassificationFeatureExtractor):
            return ClassificationFeatureSaver
        elif isinstance(obj, S3DGFeatureExtractor):
            return VideoFeatureSaver
        else:
            raise NotImplementedError


if __name__ == "__main__":
    # extract data
    video_files = [get_absolute_path("../tests/data/video/data/mp4/video9770.mp4")]
    # extractor
    s3dg = S3DGFeatureExtractor(s3d_model_path=None)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # feature manager
        fm = VideoFeatureManager(tmpdirname, "video_dataset", "train")
        fm.register_extractor(s3dg)
        fm.extract_and_save(video_files)

        feature_file = osp.join(tmpdirname, "s3dg/mixed5c/train/video9770.npy")
        assert osp.exists(feature_file)
