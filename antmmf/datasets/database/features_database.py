# Copyright (c) Facebook, Inc. and its affiliates.
from multiprocessing.pool import ThreadPool
from typing import Union

import torch
import tqdm
from dataclasses import dataclass

from antmmf.common.constants import (
    FEATURE_KEY_STR,
    FEATURE_PATH_STR,
    WRITER_STR,
    FAST_READ_STR,
    MAX_FEATURES_STR,
    DEPTH_FIRST_STR,
    RETURN_FEATURES_INFO_STR,
    FEATURE_DIM_STR,
)
from antmmf.common import AntMMFConfig
from antmmf.common.registry import registry
from antmmf.datasets.features.feature_readers import FeatureReader
from antmmf.utils.distributed import is_master
from antmmf.utils.general import get_absolute_path
from antmmf.common import Configuration


class FeaturesDatabase(torch.utils.data.Dataset):
    @dataclass
    class Config(AntMMFConfig):
        fast_read: bool = False
        depth_first: bool = False
        max_features: int = 100
        feature_dim: int = 2048  # feature dim for each image region
        return_features_info: bool = True
        only_features_info: bool = False  # load feature-info(usually bbox info) while skip loading bbox features

    def __init__(
        self,
        config: Union[Config, Configuration],
        path,
        annotation_db,
        feature_key=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.config = FeaturesDatabase.Config.create_from(config, **kwargs)
        self.feature_readers = []
        self.feature_dict = {}
        self.feature_key = self.config.get(FEATURE_KEY_STR, FEATURE_PATH_STR)
        self.feature_key = feature_key if feature_key else self.feature_key
        self._fast_read = self.config.get(FAST_READ_STR, False)
        self.writer = registry.get(WRITER_STR)

        path = path.split(",")

        for image_feature_dir in path:
            feature_reader = FeatureReader(
                base_path=get_absolute_path(image_feature_dir),
                depth_first=self.config.get(DEPTH_FIRST_STR, False),
                max_features=self.config.get(MAX_FEATURES_STR, 100),
                feature_dim=self.config.get(FEATURE_DIM_STR, 2048),
                only_features_info=self.config.only_features_info,
            )
            self.feature_readers.append(feature_reader)

        self.paths = path
        self.annotation_db = annotation_db
        assert self.annotation_db is not None, "Ensure annotation_db exists"
        self._should_return_info = config.get(RETURN_FEATURES_INFO_STR, True)

        if self._fast_read:
            self.writer.write("Fast reading features from {}".format(", ".join(path)))
            self.writer.write("Hold tight, this may take a while...")
            self._threaded_read()

    def _threaded_read(self):
        elements = [idx for idx in range(1, len(self.annotation_db))]
        pool = ThreadPool(processes=4)

        with tqdm.tqdm(total=len(elements), disable=not is_master()) as pbar:
            for i, _ in enumerate(pool.imap_unordered(self._fill_cache, elements)):
                if i % 100 == 0:
                    pbar.update(100)
        pool.close()

    def _fill_cache(self, idx):
        feat_file = self.annotation_db[idx][FEATURE_PATH_STR]
        features, info = self._read_features_and_info(feat_file)
        self.feature_dict[feat_file] = (features, info)

    def _read_features_and_info(self, feat_file):
        features = []
        infos = []
        for feature_reader in self.feature_readers:
            # feat_file is not reliable(not exist or corrupted), feature_reader is responsible
            # for handle possible exceptions.
            feature, info = feature_reader.read(feat_file)
            features.append(feature)
            infos.append(info)

        if not self._should_return_info:
            infos = None
        return features, infos

    def _get_image_features_and_info(self, feat_file):
        assert isinstance(feat_file, str)
        image_feats, infos = self.feature_dict.get(feat_file, (None, None))

        if image_feats is None:
            image_feats, infos = self._read_features_and_info(feat_file)

        return image_feats, infos

    def __len__(self):
        return len(self.annotation_db)

    def __getitem__(self, idx):
        image_info = self.annotation_db[idx]
        return self.get(image_info)

    def get(self, item):
        feature_path = item.get(self.feature_key, None)
        if feature_path is None:
            feature_path = self._get_feature_path_based_on_image(item)
        return self.from_path(feature_path)

    def from_path(self, path):
        assert isinstance(path, str)
        if "genome" in path:
            path = str(int(path.split("_")[-1].split(".")[0])) + ".npy"

        features, infos = self._get_image_features_and_info(path)

        item = {}
        for idx, image_feature in enumerate(features):
            item["image_feature_%s" % idx] = image_feature
            if infos is not None:
                # infos[idx].pop("cls_prob", None)
                item["image_info_%s" % idx] = infos[idx]

        return item

    def _get_feature_path_based_on_image(self, item):
        image_path = self._get_attrs(item)[0]
        feature_path = ".".join(image_path.split(".")[:-1]) + ".npy"
        return feature_path
