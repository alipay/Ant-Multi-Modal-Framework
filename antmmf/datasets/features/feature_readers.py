# Copyright (c) 2023 Ant Group and its affiliates.
import os
import pickle
import warnings

import numpy as np
import torch

from antmmf.common.constants import (
    MAX_FEATURES_STR,
    LMDB_EXT_STR,
    NPY_EXT_STR,
    INFO_STR,
    IMAGE_FEAT_STR,
    IMAGE_TEXT_STR,
    IMAGE_BBOX_SOURCE_STR,
    FEATURES_STR,
    IS_OCR_STR,
    IMAGE_FEATURE_STR,
)
from antmmf.utils.file_io import PathManager


class FeatureReader:
    def __init__(
        self,
        base_path,
        depth_first,
        feature_dim=None,
        ndim=2,
        max_features=None,
        only_features_info=False,
    ):
        """Feature Reader class for reading features.

        Note: Deprecation: ndim and image_feature will be deprecated later
        and the format will be standardize using features from detectron.

        Parameters
        ----------
        ndim : int
            Number of expected dimensions in features
        depth_first : bool
            CHW vs HWC
        max_features : int
            Number of maximum bboxes to keep
        only_feature_info: bool
            Whether return feature_info only, feature_info contains bbox info. AntMMF
            support feature-info npy file bbox format:

        Returns
        -------
        type
            Description of returned object.

        """
        self.base_path = base_path
        self.feat_reader = None
        self.depth_first = depth_first
        self.max_features = max_features
        self.feature_dim = feature_dim
        self.ndim = ndim
        self.only_feature_info = only_features_info

        self._init_reader()

    def _init_reader(self):
        # Currently all lmdb features are with ndim == 2
        if self.base_path.endswith(LMDB_EXT_STR):
            lmdb_reader_class = (
                LMDBFeatureInfoReader if self.only_feature_info else LMDBFeatureReader
            )
            self.feat_reader = lmdb_reader_class(
                self.max_features, self.feature_dim, self.base_path
            )
        elif self.only_feature_info:
            self.feat_reader = FeatureInfoReader()
        elif self.ndim == 2 or self.ndim == 0:
            if self.max_features is None:
                self.feat_reader = FasterRCNNFeatureReader()
            else:
                # TODO: Fix later when we move to proper standardized features
                # if isinstance(self.image_feature.item(0), dict):
                #     self.feat_reader = \
                #         PaddedFeatureRCNNWithBBoxesFeatureReader(
                #             self.max_features
                #         )
                # else:
                self.feat_reader = PaddedFasterRCNNFeatureReader(
                    self.max_features, self.feature_dim
                )
        elif self.ndim == 3 and not self.depth_first:
            self.feat_reader = Dim3FeatureReader()
        elif self.ndim == 4 and self.depth_first:
            self.feat_reader = CHWFeatureReader()
        elif self.ndim == 4 and not self.depth_first:
            self.feat_reader = HWCFeatureReader()
        else:
            raise TypeError("unkown image feature format")

    def read(self, image_feat_path):
        image_feat_path = os.path.join(self.base_path, image_feat_path)

        if self.feat_reader is None:
            # Currently all lmdb features are with ndim == 2 so we are
            # avoiding loading the lmdb to determine feature ndim
            self._init_reader()

        return self.feat_reader.read(image_feat_path)


class FeatureInfoReader:
    def read(self, image_feat_path):
        info_path = "{}_info.npy".format(image_feat_path.split(NPY_EXT_STR)[0])
        image_info = {}
        if PathManager.exists(info_path):
            try:
                image_info = np.load(info_path, allow_pickle=True).item()
            except (OSError, ValueError):
                warnings.warn(f"Corrupt npy feature-info file:{info_path}")
                image_info = {}
        return None, image_info


class FasterRCNNFeatureReader:
    def read(self, image_feat_path):
        return torch.from_numpy(np.load(image_feat_path)), None


class CHWFeatureReader:
    def read(self, image_feat_path):
        feat = np.load(image_feat_path)
        assert feat.shape[0] == 1, "batch is not 1"
        feat = torch.from_numpy(feat.squeeze(0))
        return feat, None


class Dim3FeatureReader:
    def read(self, image_feat_path):
        tmp = np.load(image_feat_path)
        _, _, c_dim = tmp.shape
        image_feature = torch.from_numpy(np.reshape(tmp, (-1, c_dim)))
        return image_feature, None


class HWCFeatureReader:
    def read(self, image_feat_path):
        tmp = np.load(image_feat_path)
        assert tmp.shape[0] == 1, "batch is not 1"
        _, _, _, c_dim = tmp.shape
        image_feature = torch.from_numpy(np.reshape(tmp, (-1, c_dim)))
        return image_feature, None


class PaddedFasterRCNNFeatureReader:
    def __init__(self, max_loc, feat_dim):
        self.max_loc = max_loc
        self.feat_dim = feat_dim
        self.first = True
        self.take_item = False

    def feature_io(self, image_feat_path):
        try:
            image_feature = np.load(image_feat_path, allow_pickle=True)
        except (OSError, ValueError, FileNotFoundError):
            warnings.warn(f"Corrupt npy feature file:{image_feat_path}")
            image_feature = np.zeros((self.max_loc, self.feat_dim), dtype=np.float32)

        info_path = "{}_info.npy".format(image_feat_path.split(NPY_EXT_STR)[0])
        image_info = {}

        if PathManager.exists(info_path):
            try:
                info_path = np.load(info_path, allow_pickle=True).item()
            except (OSError, ValueError):
                warnings.warn(f"Corrupt npy feature-info file:{info_path}")
                info_path = {}
            image_info.update(info_path)
        return image_feature, image_info

    def prepare_feature(self, image_feature, image_info):
        if image_feature is None:
            # image_feature is always None for LMDBFeatureInfoReader
            # image_feature is padded empty numpy-arrays for LMDBFeatureReader
            return None, image_info

        if self.first:
            self.first = False
            if image_feature.size == 1 and IMAGE_FEAT_STR in image_feature.item():
                self.take_item = True

        if self.take_item:
            item = image_feature.item()
            if IMAGE_TEXT_STR in item:
                image_info[IMAGE_TEXT_STR] = item[IMAGE_TEXT_STR]
                image_info[IS_OCR_STR] = item[IMAGE_BBOX_SOURCE_STR]
                image_feature = item[IMAGE_FEAT_STR]

            if INFO_STR in item:
                if IMAGE_TEXT_STR in item[INFO_STR]:
                    image_info.update(item[INFO_STR])
                image_feature = item[FEATURES_STR]

        # Handle the case of ResNet152 features
        if len(image_feature.shape) > 2:
            shape = image_feature.shape
            image_feature = image_feature.reshape(-1, shape[-1])

        image_loc, image_dim = image_feature.shape
        if self.feat_dim is not None:
            assert image_dim == self.feat_dim
        tmp_image_feat = np.zeros((self.max_loc, image_dim), dtype=np.float32)
        tmp_image_feat[
            0:image_loc,
        ] = image_feature[: self.max_loc, :]
        image_feature = torch.from_numpy(tmp_image_feat)

        image_info[MAX_FEATURES_STR] = torch.tensor(image_loc, dtype=torch.long)
        return image_feature, image_info

    def read(self, image_feat_path):
        image_feature, image_info = self.feature_io(image_feat_path)
        return self.prepare_feature(image_feature, image_info)


class LMDBFeatureReader(PaddedFasterRCNNFeatureReader):
    def __init__(self, max_loc, feat_dim, base_path):
        super().__init__(max_loc, feat_dim)
        self.db_path = base_path

        if not os.path.exists(self.db_path):
            raise RuntimeError(
                "{} path specified for LMDB features doesn't exists.".format(
                    self.db_path
                )
            )
        self._init_db()

    def _init_db(self):
        import lmdb

        self.env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False, buffers=True) as txn:
            self.image_ids = pickle.loads(txn.get(b"keys"))
            self.image_id_indices = {
                self.image_ids[i]: i for i in range(0, len(self.image_ids))
            }

    def feature_io(self, image_file_path):
        if self.env is None:
            self._init_db()

        default_image_feature = np.zeros(
            (self.max_loc, self.feat_dim), dtype=np.float32
        )
        default_image_info = {}
        split = os.path.relpath(image_file_path, self.db_path).split(NPY_EXT_STR)[0]
        if not split:
            warnings.warn(f"skip loading feature:{image_file_path}")
            return default_image_feature, default_image_info

        try:
            image_id = int(split.split("_")[-1])
        except (ValueError, KeyError):
            # The image id is complex or involves folder, use it directly
            image_id = str(split).encode()

        with self.env.begin(write=False, buffers=True) as txn:
            bytes_image_info = txn.get(image_id)
            if bytes_image_info is not None:
                image_info = pickle.loads(bytes_image_info)
            else:
                image_info = {}
                warnings.warn(f"feature not found in lmdb:{image_id}")

        image_feature = image_info.get(FEATURES_STR, default_image_feature)
        return image_feature, image_info


class LMDBFeatureInfoReader(LMDBFeatureReader):
    """
    LMDB reader for loading feature info(bbox only)
    """

    def feature_io(self, image_file_path):
        if self.env is None:
            self._init_db()

        default_image_info = {}
        split = os.path.relpath(image_file_path, self.db_path).split(NPY_EXT_STR)[0]
        if not split:
            warnings.warn(f"skip loading feature:{image_file_path}")
            return None, default_image_info
        try:
            image_id = int(split.split("_")[-1])
        except (ValueError, KeyError):
            # The image id is complex or involves folder, use it directly
            image_id = str(split).encode()

        with self.env.begin(write=False, buffers=True) as txn:
            bytes_image_info = txn.get(image_id)
            if bytes_image_info is not None:
                image_info = pickle.loads(bytes_image_info)
            else:
                image_info = {}
                warnings.warn(f"feature not found in lmdb:{image_id}")
        return None, image_info


class PaddedFeatureRCNNWithBBoxesFeatureReader:
    def __init__(self, max_loc):
        self.max_loc = max_loc

    def read(self, image_feat_path):
        image_feat_bbox = np.load(image_feat_path)
        image_boxes = image_feat_bbox.item().get("image_bboxes")
        tmp_image_feat = image_feat_bbox.item().get(IMAGE_FEATURE_STR)
        image_loc, image_dim = tmp_image_feat.shape
        tmp_image_feat_2 = np.zeros((self.max_loc, image_dim), dtype=np.float32)
        tmp_image_feat_2[
            0:image_loc,
        ] = tmp_image_feat
        tmp_image_feat_2 = torch.from_numpy(tmp_image_feat_2)
        tmp_image_box = np.zeros((self.max_loc, 4), dtype=np.int32)
        tmp_image_box[0:image_loc] = image_boxes
        tmp_image_box = torch.from_numpy(tmp_image_box)
        image_info = {
            "image_bbox": tmp_image_box,
            MAX_FEATURES_STR: torch.tensor(image_loc, dtype=torch.int),
        }

        return tmp_image_feat_2, image_info
