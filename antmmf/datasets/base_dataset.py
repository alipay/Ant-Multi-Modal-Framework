# Copyright (c) 2023 Ant Group and its affiliates.
import os
from typing import Dict, Union

from torch.utils.data.dataset import Dataset, IterableDataset

from antmmf.common.constants import DATASETS_STR, FEATURES_STR, USE_FEATURE_STR
from antmmf.common.registry import registry
from antmmf.structures import SampleList
from antmmf.datasets.database.annotated import AnnotatedDatabase
from antmmf.datasets.database.features_database import FeaturesDatabase
from antmmf.datasets.database.image_database import ImageDatabase
from antmmf.datasets.database.video_database import VideoClipsDatabase
from antmmf.datasets.processors.processors import Processor
from antmmf.common.configuration import get_zoo_config
from antmmf.utils.download import download_based_on_config
from antmmf.utils.general import get_antmmf_root
from antmmf.common import Configuration


def _get_config_path(config, sec, index):
    if isinstance(config[sec], list) and index >= 0:
        return config[sec][index]
    else:
        return config[sec]


class MMFDataset:
    def __init__(
        self, name: str, dataset_type: str, config: Union[Configuration, Dict]
    ):
        self._name = name
        self._dataset_type = dataset_type
        self.config = config

    @property
    def dataset_type(self):
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, dataset_type):
        self._dataset_type = dataset_type

    @property
    def name(self):
        return self._name

    def init_processors(self):
        if not hasattr(self.config, "processors"):
            return
        extra_params = {"data_root_dir": self.config.data_root_dir}

        for processor_key, processor_params in self.config.processors.items():
            reg_key = "{}_{}".format(self._name, processor_key)
            reg_check = registry.get(reg_key, no_warning=True)
            # Note: processors will be initialized only once
            if reg_check is None:
                processor_object = Processor(processor_params, **extra_params)
                setattr(self, processor_key, processor_object)
                registry.register(reg_key, processor_object)
            else:
                setattr(self, processor_key, reg_check)

    def format_for_evalai(self, report):
        return []

    def align_evalai_report_order(self, report):
        return report

    def verbose_dump(self, *args, **kwargs):
        return

    def try_fast_read(self):
        return

    def prepare_batch(self, batch):
        """
        Can be possibly overriden in your child class

        Prepare batch for passing to model. Whatever returned from here will
        be directly passed to model's forward function. Currently moves the batch to
        proper device.

        Args:
            batch (SampleList): sample list containing the currently loaded batch

        Returns:
            sample_list (SampleList): Returns a sample representing current batch loaded
        """
        # Should be a SampleList
        if not isinstance(batch, SampleList):
            # Try converting to SampleList
            batch = SampleList(batch)
        batch = batch.to(self._device)
        return batch

    def get_annotations(self, attr):
        """
        Return the corresponding `attr` attribute of the whole dataset as a list.
        This method provides an easy way for access datasets' attrs, probably used in sampling phase.
        Args:
            attr: attributes of annotations

        Returns:

        """
        raise NotImplementedError


class BaseDataset(MMFDataset, Dataset):
    """Base class for implementing a dataset. Inherits from PyTorch's Dataset class
    but adds some custom functionality on top. Instead of ``__getitem__`` you have to implement
    ``get_item`` here. Processors mentioned in the configuration are automatically initialized for
    the end user.

    Args:
        name (str): Name of your dataset to be used a representative in text strings
        dataset_type (str): Type of your dataset. Normally, train|val|test
        config (Configuration): Configuration for the current dataset
    """

    def __init__(self, name, dataset_type, config={}, index=-1, *args, **kwargs):
        super(BaseDataset, self).__init__(name, dataset_type, config)
        self._device = registry.get("current_device")
        self.use_cuda = "cuda" in str(self._device)
        self._index = index  # 如果是-1，则没有设置数据集index

        self._download_requirement(config, name)

        self.annotation_db = self._build_annotation_db()
        self.image_db = self._build_image_db()
        self.video_db = self._build_video_db()
        self.feature_db = self._build_features_db()

        self.init_processors()
        self.setup_extras(dataset_type, config, *args, **kwargs)

    def setup_extras(self, dataset_type, config, *args, **kwargs):
        return

    def _download_requirement(
        self, config, requirement_key, requirement_variation="defaults"
    ):
        version, resources = get_zoo_config(
            requirement_key, requirement_variation, None, DATASETS_STR
        )

        if resources is None:
            return

        if config.get(USE_FEATURE_STR, False) is False:
            return

        flist = config.features[self._dataset_type]
        if isinstance(flist, str):
            flist = [flist]
        for res_file in flist:
            res_file = self._get_absolute_path(res_file)
            if os.path.exists(res_file):
                continue

            download_based_on_config(
                config,
                resources,
                # need to go one directory up because this function will
                # append attribute such as features or models to the directory
                os.path.dirname(res_file) + "/..",
                version,
            )

    def _build_annotation_db(self, database_cls=AnnotatedDatabase, **kwargs):
        if "annotations" in self.config:
            annotation_path = _get_config_path(
                self.config["annotations"], self._dataset_type, self._index
            )
            annotation_path = self._get_absolute_path(annotation_path)
            return database_cls(annotation_path, **kwargs)
        else:
            return None

    def _build_image_db(self, database_cls=ImageDatabase, **kwargs):
        if self.config.get("use_images", False):
            assert (
                "images" in self.config
            ), "use images is set but images are not specified"
            image_path = _get_config_path(
                self.config["images"], self._dataset_type, self._index
            )
            image_path = self._get_absolute_path(image_path)
            return database_cls(image_path, annotation_db=self.annotation_db, **kwargs)
        else:
            return None

    def _build_video_db(self, database_cls=VideoClipsDatabase, **kwargs):
        if self.config.get("use_videos", False):
            assert (
                "videos" in self.config
            ), "use videos is set but videos are not specified"
            video_path = _get_config_path(
                self.config["videos"], self._dataset_type, self._index
            )
            video_path = self._get_absolute_path(video_path)
            return database_cls(video_path, annotation_db=self.annotation_db, **kwargs)
        else:
            return None

    def _build_features_db(self, database_cls=FeaturesDatabase, **kwargs):
        if self.config.get(USE_FEATURE_STR, False):
            features_path = _get_config_path(
                self.config[FEATURES_STR], self._dataset_type, self._index
            )
            features_path = self._get_absolute_path(features_path)
            return database_cls(
                self.config, features_path, annotation_db=self.annotation_db, **kwargs
            )
        else:
            return None

    def _get_absolute_path(self, paths):
        if isinstance(paths, list):
            return [self._get_absolute_path(path) for path in paths]
        elif isinstance(paths, str):
            if not os.path.isabs(paths):
                maybe_root_dir = os.path.join(os.getcwd(), self.config.data_root_dir)
                # when data_root_dir is abs_path, don't need `data_root_dir` prefix
                if os.path.isabs(self.config.data_root_dir):
                    data_root_dir = ""
                elif os.path.exists(maybe_root_dir) and os.path.isdir(maybe_root_dir):
                    data_root_dir = os.getcwd()
                else:
                    data_root_dir = get_antmmf_root()  # make it compatible to old codes
                paths = os.path.join(data_root_dir, self.config.data_root_dir, paths)
            return paths
        else:
            raise TypeError("Paths passed to dataset should either be string or list")

    def get_item(self, index):
        """
        override in your dataset first
        """
        raise NotImplementedError(
            f"get_item is not implemented in {type(self)}, pls. overwrite it in your dataset."
        )

    def __getitem__(self, idx):
        # TODO: Add warning about overriding
        """
        Internal __getitem__. Don't override, instead override ``get_item`` for your usecase.

        .. warning::

            DO NOT OVERRIDE in child class. Instead override ``get_item``.
        """
        sample = self.get_item(idx)
        if sample is None:
            return None

        sample.dataset_type = self._dataset_type
        sample.dataset_name = self._name
        return sample


class BaseIterableDataset(MMFDataset, IterableDataset):
    """Base class for implementing a iterable dataset. Inherits from PyTorch's IterableDataset class
    but adds some custom functionality on top.

    Args:
        name (str): Name of your dataset to be used a representative in text strings
        dataset_type (str): Type of your dataset. Normally, train|val|test
        config (Configuration): Configuration for the current dataset
    """

    def __init__(self, name, dataset_type, config={}, *args, **kwargs):
        super().__init__(name, dataset_type, config)
        self._device = registry.get("current_device")
        self.use_cuda = "cuda" in str(self._device)
        self.iter_db = self._build_iter_db()

        self.init_processors()

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
