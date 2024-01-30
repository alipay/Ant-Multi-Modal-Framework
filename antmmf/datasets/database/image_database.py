# Copyright (c) 2023 Ant Group and its affiliates.
import collections
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor

import PIL.Image
import numpy as np
import torch
import torchvision
import torchvision.datasets.folder as tv_helpers

from antmmf.utils.file_io import PathManager
from antmmf.utils.general import flatten_list
from antmmf.utils.general import get_absolute_path

# Disable PIL.Image.DecompressionBombError, assume users are responsible for trustworthy image sources.
# See detail at: https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
PIL.Image.MAX_IMAGE_PIXELS = None


def get_possible_image_paths(path):
    for ext in tv_helpers.IMG_EXTENSIONS:
        image_ext = ".".join(path.split(".")[:-1]) + ext
        if PathManager.isfile(image_ext):
            path = image_ext
            break
    return path


class ImageDatabase(torch.utils.data.Dataset):
    """ImageDatabase can be used to load images in MMF.
    This goes either in conjunction with AnnotationDatabase or
    can be separately used with function such as `from_path`.
    MMFDataset initializes its own copy of ImageDatabase if `use_images`
    is True. Rest everything works same as a normal torch Dataset if
    you pass the annotation_db as a parameter. For example for item
    1 from annotation db, you can pass same id to ImageDatabase to loads
    its image. If you don't pass it, you have two options. Either use
    .get which takes in an annotation db item or .from_path which directly
    takes in an image path. You are free to use your own dataset instead
    of image database or free to update or ignore MMFDataset's ImageDataset
    initialization. You can either reinitialize with transform and other
    params or use any of torchvision's datasets.
    """

    DEFAULT_LOADING_WORKERS = 1

    def __init__(
        self,
        path,
        annotation_db=None,
        transform=None,
        loader=tv_helpers.default_loader,
        is_valid_file=None,
        image_field_keys=None,
        *args,
        **kwargs,
    ):
        """Initialize an instance of ImageDatabase

        Args:
            torch ([type]): [description]
            config (DictConfig): Config object from dataset_config
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
            image_field_keys(list, optional): indicate image keys of annotation_db to load.
                If image_field_keys not indicated, using `_get_attrs` method to get possible
                images to load.
        """
        super().__init__()
        self.base_path = get_absolute_path(path)
        if isinstance(self.base_path, list):
            assert len(self.base_path) == 1, "only supprt one base path"
            self.base_path = self.base_path[0]
        self._transform = transform
        self._annotation_db = annotation_db
        self.loader = loader
        self.is_valid_file = is_valid_file
        self.image_field_keys = image_field_keys
        self.kwargs = kwargs
        self.executor = ThreadPoolExecutor(
            max_workers=ImageDatabase.DEFAULT_LOADING_WORKERS
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

    def __len__(self):
        self._check_annotation_db_present()
        return len(self.annotation_db)

    def __getitem__(self, idx):
        self._check_annotation_db_present()
        item = self.annotation_db[idx]
        return self.get(item)

    def _check_annotation_db_present(self):
        if not self.annotation_db:
            raise AttributeError(
                "'annotation_db' must be set for the database to use __getitem__."
                + " Use image_database.annotation_db to set it."
            )

    def get(self, item):
        if self.image_field_keys is None:
            possible_images = self._get_attrs(item)
        else:
            possible_images = [item.get(img_key) for img_key in self.image_field_keys]
        # flatten it if possible
        images_to_load = flatten_list(possible_images)
        # filter images you do not want to load
        if self.is_valid_file is not None and callable(self.is_valid_file):
            images_to_load = [img for img in images_to_load if self.is_valid_file(img)]

        images_mask = np.ones(len(images_to_load), dtype=np.int32)

        max_image_num = self.kwargs.get("num_images_of_each_sample")
        if max_image_num is not None:
            images_mask = np.zeros(max_image_num, dtype=np.int32)
            if len(images_to_load) >= max_image_num:
                # Sequentially sampling max_image_num images, which help adapt to various fps sampling strategies in
                # practical. But bring various time granularity for each image token
                idxes = list(range(len(images_to_load)))
                images_to_load = [
                    images_to_load[idx]
                    for idx in sorted(random.sample(idxes, max_image_num))
                ]
                images_mask[:] = 1
            else:
                num_padding = max_image_num - len(images_to_load)
                images_to_load += [None] * num_padding
                images_mask[:-num_padding] = 1
        return self.from_path(images_to_load, images_mask)

    def from_path(self, paths, images_mask, use_transforms=True):
        if isinstance(paths, str):
            paths = [paths]

        assert isinstance(
            paths, collections.abc.Iterable
        ), "Path needs to a string or an iterable"

        def load_func(image_path):
            if image_path is None:
                return None, None

            if not os.path.isabs(image_path):
                image_path = os.path.join(self.base_path, image_path)

            if not PathManager.isfile(image_path):
                image_path = get_possible_image_paths(image_path)

            if not PathManager.exists(image_path):
                # skip images that are not exist
                warnings.warn(
                    "Image not found at path {}.{{jpeg|jpg|svg|png}}.".format(
                        ".".join(image_path.split(".")[:-1])
                    )
                )
                return None, None

            try:
                image = self.open_image(image_path)
            except (OSError, NameError):
                warnings.warn(f"Corrupted image:{image_path}")
                return None, None

            image_height, image_width = np.array(image).shape[:2]

            if self.transform and use_transforms:
                image = self.transform(image)
            return image, (image_height, image_width)

        # parallel loading image
        images_infos = list(self.executor.map(load_func, paths))
        return {
            "images": [x[0] for x in images_infos],
            "images_mask": images_mask,
            "image_shape": [x[1] for x in images_infos],
        }

    def open_image(self, path):
        return self.loader(path)

    def _get_attrs(self, item):
        """Returns possible attribute that can point to image id

        Args:
            item (Object): Object from the DB

        Returns:
            List[str]: List of possible images that will be copied later
        """
        image = None
        pick = None
        attrs = self._get_possible_attrs()

        for attr in attrs:
            image = item.get(attr, None)
            if image is not None:
                pick = attr
                break

        # Check if first one is nlvr2
        if pick == "identifier" and "left_url" in item and "right_url" in item:
            return [image + "-img0.jpg", image + "-img1.jpg"]
        elif pick == "image_name" or pick == "image_id":
            return [image + ".jpeg"]
        else:
            return [image]

    def _get_possible_attrs(self):
        return [
            "Flickr30kID",
            "Flikr30kID",
            "identifier",
            "image_path",
            "image_name",
            "image",  # datacube image path
            "img",
            "image_id",
        ]
