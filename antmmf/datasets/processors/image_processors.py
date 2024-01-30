# Copyright (c) 2023 Ant Group and its affiliates.
"""
The processors exist in antmmf to make data processing pipelines in various
datasets as similar as possible while allowing code reuse.

The processors also help maintain proper abstractions to keep only what matters
inside the dataset's code. This allows us to keep the dataset ``get_item``
logic really clean and no need about maintaining opinions about data type.
Processors can work on both images and text due to their generic structure.

To create a new processor, follow these steps:

1. Inherit the ``BaseProcessor`` class.
2. Implement ``_call`` function which takes in a dict and returns a dict with
   same keys preprocessed as well as any extra keys that need to be returned.
3. Register the processor using ``@registry.register_processor('name')`` to
   registry where 'name' will be used to refer to your processor later.

In processor's config you can specify ``preprocessor`` option to specify
different kind of preprocessors you want in your dataset.

Let's break down processor's config inside a dataset (VQA2.0) a bit to understand
different moving parts.

Config::

    task_attributes:
        vqa:
            dataset_attributes:
                vqa2:
                    processors:
                      text_processor:
                        type: vocab
                        params:
                          max_length: 14
                          vocab:
                            type: intersected
                            embedding_name: glove.6B.300d
                            vocab_file: vocabs/vocabulary_100k.txt
                          answer_processor:
                            type: vqa_answer
                            params:
                              num_answers: 10
                              vocab_file: vocabs/answers_vqa.txt
                              preprocessor:
                                type: simple_word
                                params: {}

``BaseDataset`` will init the processors and they will be available inside your
dataset with same attribute name as the key name, e.g. `text_processor` will
be available as `self.text_processor` inside your dataset. As is with every module
in antmmf, processor also accept a ``Configuration`` with a `type` and `params`
attributes. `params` defined the custom parameters for each of the processors.
By default, processor initialization process will also init `preprocessor` attribute
which can be a processor config in itself. `preprocessor` can be then be accessed
inside the processor's functions.

Example::

    from antmmf.common.registry import registry
    from antmmf.tasks.processors import BaseProcessor


    class MyProcessor(BaseProcessor):
        def __init__(self, config, *args, **kwargs):
            return

        def __call__(self, item, *args, **kwargs):
            text = item['text']
            text = [t.strip() for t in text.split(" ")]
            return {"text": text}
"""
import collections.abc
import inspect
import random
from dataclasses import dataclass
from typing import List, Union, Dict
from typing import Optional

import numpy as np
import torch
import torchvision
from PIL import Image

from antmmf.common.configuration import Configuration, AntMMFConfig
from antmmf.common.registry import registry
from antmmf.datasets.processors.transforms import detection as T
from antmmf.datasets.processors.processors import BaseProcessor
from antmmf.datasets.processors.text_processors import VocabProcessor
from antmmf.utils import dataset_utils
from antmmf.utils import image_ops
from antmmf.utils.general import AttrDict
from antmmf.utils.general import check_required_keys
from antmmf.utils.image_ops import (
    GroupScale,
    GroupCenterCrop,
    Stack,
    ToTorchFormatTensor,
    GroupNormalize,
    IdentityTransform,
    ImageLoader,
    GroupMultiScaleCrop,
    GroupRandomHorizontalFlip,
)


@registry.register_processor("bbox")
class BBoxProcessor(VocabProcessor):
    """Generates bboxes in proper format.
    Takes in a dict which contains "info" key which is a list of dicts
    containing following for each of the the bounding box

    Example bbox input::

        {
            "info": [
                {
                    "bounding_box": {
                        "top_left_x": 100,
                        "top_left_y": 100,
                        "width": 200,
                        "height": 300
                    }
                },
                ...
            ]
        }


    This will further return a Sample in a dict with key "bbox" with last
    dimension of 4 corresponding to "xyxy". So sample will look like following:

    Example Sample::

        Sample({
            "coordinates": torch.Size(n, 4),
            "width": List[number], # size n
            "height": List[number], # size n
            "bbox_types": List[str] # size n, either xyxy or xywh.
            # currently only supports xyxy.
        })

    """

    def __init__(self, config, *args, **kwargs):
        from antmmf.utils.dataset_utils import build_bbox_tensors

        self.lambda_fn = build_bbox_tensors
        self._init_extras(config)

    def __call__(self, item):
        info = item["info"]
        if self.preprocessor is not None:
            info = self.preprocessor(info)

        return {"bbox": self.lambda_fn(info, self.max_length)}


kOutputSize = 224


@registry.register_processor("normalized_image")
class NormImageProcessor(BaseProcessor):
    """Use image with normalization when you have image file and you want to extract
    features from it.

    **Key**: image

    Example Config::

        task_attributes:
            vqa:
                vqa2:
                    processors:
                      image_processor:
                        type: normalized_image
                        params:
                          input_size: None (default 224)
                          basemodel: {resnet}
                          new_length: 1
                          mode: {train, test, val}
                          image:
                            modality: {RGB, Flow, RGBDiff}
                            root_path: path to image dir
                            new_length: None (default 1 if RGB else 5)

    Args:
        config (Configuration): node containing configuration parameters of
                             the processor

    """

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self._init_extras(config, args, kwargs)
        self._image_loader = None
        # mujian: preprocessor may behave differently during training & test phrase,
        # add test_mode to support this scenario, default as
        # test for compatity
        self._mode = None
        self.set_mode(config.get("mode", "test"))

    @property
    def preprocessor(self):
        if self._mode == "train":
            return self.train_preprocessor
        else:
            return self.test_preprocessor

    def set_mode(self, mode):
        """
        dynamically choosing preprocessors during runtime
        """
        assert mode in ["train", "val", "test"]
        self._mode = mode

    @property
    def scale_size(self):
        return self.input_size * 256 // kOutputSize

    @property
    def crop_size(self):
        return self.input_size

    def _init_extras(self, config, *args, **kwargs):
        self.writer = registry.get("writer")

        # Data loading code
        self.modality = config.get("modality")
        self.base_model = config.get("basemodel")
        new_length = config.get("new_length")
        if new_length is None:
            self.new_length = 1 if self.modality == "RGB" else 5
        else:
            self.new_length = new_length

        self.input_size = config.get("input_size", kOutputSize)
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        if "resnet" in self.base_model:
            if self.modality == "Flow":
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == "RGBDiff":
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = (
                    self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
                )
        if self.modality != "RGBDiff":
            normalize = GroupNormalize(self.input_mean, self.input_std)
        else:
            normalize = IdentityTransform()

        train_transform_aug = self._get_augmentation(flip=True)
        test_transform_aug = torchvision.transforms.Compose(
            [GroupScale(int(self.scale_size)), GroupCenterCrop(self.crop_size)]
        )

        self.train_preprocessor = torchvision.transforms.Compose(
            [
                train_transform_aug,
                Stack(roll=self.base_model in ["BNInception", "InceptionV3"]),
                ToTorchFormatTensor(
                    div=self.base_model not in ["BNInception", "InceptionV3"]
                ),
                normalize,
            ]
        )

        self.test_preprocessor = torchvision.transforms.Compose(
            [
                test_transform_aug,
                Stack(roll=self.base_model in ["BNInception", "InceptionV3"]),
                ToTorchFormatTensor(
                    div=self.base_model not in ["BNInception", "InceptionV3"]
                ),
                normalize,
            ]
        )

    def _get_augmentation(self, flip=True):
        if self.modality == "RGB":
            if flip:
                return torchvision.transforms.Compose(
                    [
                        GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75, 0.66]),
                        GroupRandomHorizontalFlip(is_flow=False),
                    ]
                )
            else:
                return torchvision.transforms.Compose(
                    [GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75, 0.66])]
                )
        elif self.modality == "Flow":
            return torchvision.transforms.Compose(
                [
                    GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75]),
                    GroupRandomHorizontalFlip(is_flow=True),
                ]
            )
        elif self.modality == "RGBDiff":
            return torchvision.transforms.Compose(
                [
                    GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75]),
                    GroupRandomHorizontalFlip(is_flow=False),
                ]
            )
        else:
            raise TypeError("Unknown modality:{}".format(self.modality))

    def __call__(self, item):
        """Call requires item to have either "image" attribute or either
        "feature" attribute. If "image" is present, it will processed using
        an image processor.

        Args:
            item (Dict): Dict containing the
            "image" : the diretory of the image files to be processed
            "index" the image index,
            the exact image file is refereed as <image>/img_0000<index>.jpg if
            using a template of img_{:05d}.jpg

            or
            "feature" the processed image

        Returns:
            Dict: Dict containing indices in "feature" key, "image" in "image"
                  key and "index" of the image.

        """
        if not isinstance(item, dict):
            raise TypeError(
                "Argument passed to the processor must be a dict with either 'image' or 'feature' as keys"
            )
        obs = item.get("image")
        if "feature" in item:
            # has done the featurization
            feature = item["feature"]
        elif "image" in item:
            if self.preprocessor is None:
                raise AssertionError(
                    "If feature are not provided, an image or video_capture processor must be defined in the config"
                )
            # lazy initialize ImageLoader, for only item with 'image' field
            # uses it
            if self._image_loader is None:
                if not hasattr(self.config, "image"):
                    raise AttributeError(
                        "config passed to the processor has no attribute image, which specify the directory with images"
                    )
                self._image_loader = ImageLoader(**self.config.image)

            obs = self._image_loader.load_image(item["image"], item.get("index", 0))
            # Group Transformer extract a list of images
            feature = self.preprocessor(obs)
        elif "video_capture" in item:
            if self.preprocessor is None:
                raise AssertionError(
                    "If feature are not provided, an image or video_capture processor must be defined in the config"
                )

            obs = item["video_capture"]
            feature = self.preprocessor(obs)
        else:
            raise AssertionError(
                "A dict with either 'feature' 'image' or 'video_capture' keys must be passed to the processor"
            )

        ret = {"image": obs, "index": item.get("index", 0), "feature": feature}
        return ret


@registry.register_processor("torchvision_transforms")
class TorchvisionTransforms(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        transform_params = config.transforms
        assert isinstance(transform_params, dict) or isinstance(transform_params, list)
        if isinstance(transform_params, dict):
            transform_params = [transform_params]

        transforms_list = []

        for param in transform_params:
            if isinstance(param, collections.abc.Mapping):
                # This will throw config error if missing
                transform_type = param["type"]
                transform_param = param.get("params", {})
            else:
                assert isinstance(param, str), (
                    "Each transform should either be str or dict containing "
                    + "type and params"
                )
                transform_type = param
                transform_param = []

            transform = getattr(torchvision.transforms, transform_type, None)
            # If torchvision doesn't contain this, check our registry if we
            # implemented a custom transform as processor
            if transform is None:
                transform = registry.get_processor_class(transform_type)
            assert (
                transform is not None
            ), f"torchvision.transforms has no transform {transform_type}"

            # https://github.com/omry/omegaconf/issues/248
            # transform_param = OmegaConf.to_container(transform_param)
            # If a dict, it will be passed as **kwargs, else a list is *args
            if isinstance(transform_param, collections.abc.Mapping):
                transform_object = transform(**transform_param)
            else:
                transform_object = transform(*transform_param)

            transforms_list.append(transform_object)

        self.transform = torchvision.transforms.Compose(transforms_list)

    def __call__(self, x):
        # Support both dict and normal mode
        if isinstance(x, collections.abc.Mapping):
            x.update({"image": self.transform(x["image"])})
            return x
        else:
            return self.transform(x)


@registry.register_processor("GrayScaleTo3Channels")
class GrayScaleTo3Channels(BaseProcessor):
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, x):
        if isinstance(x, collections.abc.Mapping):
            x.update({"image": self.transform(x["image"])})
            return x
        else:
            return self.transform(x)

    def transform(self, x):
        assert isinstance(x, torch.Tensor)
        # Handle grayscale, tile 3 times
        if x.size(0) == 1:
            x = torch.cat([x] * 3, dim=0)
        return x


@registry.register_processor("custom_transforms")
class CustomTransforms(BaseProcessor):
    """
    Processor wrapper for image augmentation functions defined in antmmf.utils.dataset_utils
    or antmmf.utils.image_ops
    Example Config::
        processor:
          type: custom_transforms
          params:
            mode: random # or sequential
            transforms: # define augmentation function list to use
              - type: random_crop # function name
                params: # function params
                  four_side_ratios: [0.5, 0.2, 0.1]
              - type: random_black
              - type: random_flip
              - type: random_affine2
              - type: random_color2
              - type: random_noise
              - type: random_blur
              - type: random_gray
              - type: randLogo
                params:
                  random_logo_dir: path_to_logo_dir
              - type: randText
              - type: pictureInPicture
                params:
                  picture_dir: path_to_picture_dir

    """

    def __init__(self, config, *args, **kwargs):
        self.config = config
        assert self.config.mode in ["sequential", "random"]
        transform_types = config.transforms
        assert isinstance(transform_types, list)

        self.transfunc_list, self.transfunc_params = [], []

        for tranform in transform_types:
            assert isinstance(tranform, (dict, Configuration))
            transform_type = tranform["type"]
            transform_param = tranform.get("params", {})
            # get transform_obj from antmmf.utils.dataset_utils
            transform_obj = getattr(dataset_utils, transform_type, None)
            if transform_obj is None:  # get transform_obj from antmmf.utils.image_ops
                transform_obj = getattr(image_ops, transform_type, None)
            if (
                transform_obj is None
            ):  # roll back to transform_obj from torchvision.transform
                transform_obj = getattr(torchvision.transforms, transform_type, None)

            assert (
                transform_obj is not None
            ), f"antmmf.utils.[dataset_utils|image_ops] has no transform: {transform_type}"

            if inspect.isfunction(transform_obj):
                self.transfunc_list.append(transform_obj)
                self.transfunc_params.append(transform_param)
            elif inspect.isclass(transform_obj):
                if isinstance(transform_param, collections.abc.Mapping):
                    transform_instance = transform_obj(**transform_param)
                else:
                    transform_instance = transform_obj(*transform_param)
                assert callable(
                    transform_instance
                ), f"class {transform_type} in antmmf.utils.[dataset_utils|image_ops] is not callable"
                self.transfunc_list.append(transform_instance)
                self.transfunc_params.append(transform_param)
            else:
                raise Exception(
                    "Cannot recognized transform:{transform_type} defined in antmmf.utils.[dataset_utils|image_ops]"
                )

    def __call__(self, x):
        return_dict = False
        # Support both dict and normal mode
        if isinstance(x, collections.abc.Mapping):
            x = x["image"]
            return_dict = True
        # transform needs float32 format
        if x.dtype == torch.uint8:
            x = x.float()

        if self.config.mode == "sequential":
            res = self.sequential_transform(x)
            idx = None
        else:
            res, idx = self.random_transform(x)

        if return_dict:
            res = {"image": res}
            if idx is not None:
                res.update({"idx": idx})
        return res

    def sequential_transform(self, x):
        for func, param in zip(self.transfunc_list, self.transfunc_params):
            x = func(x, **param)
        return x

    def random_transform(self, x):
        num_transforms = len(self.transfunc_list)
        idx = self.config.get("idx", None)
        if idx is None:
            idx = random.randint(0, num_transforms)
        if idx < num_transforms:
            x = self.transfunc_list[idx](x, **self.transfunc_params[idx])
        if type(x) is np.ndarray:
            x = Image.fromarray(x, mode="RGB")
        return x, idx


@registry.register_processor("random_flip_processor")
class RandomFlipProcessor(BaseProcessor):
    def __init__(self, *args, **config):
        config = AttrDict(config)
        self.prob = config.prob

    def __call__(self, data):
        """
        Args:
            data: {'image': image_arr,
                   'labels': annotations, # Nx5},
                   image_arr is BGR or RGB format,
                   labels: Lx1y1x2y2, unnormalized

        Returns:
        """
        if random.random() < self.prob:
            data["image"] = np.fliplr(data["image"])
            if "labels" in data:
                height, width = data["image"].shape[:2]
                x1, x2 = np.copy(data["labels"][:, 1]), np.copy(data["labels"][:, 3])
                data["labels"][:, 1] = width - x2
                data["labels"][:, 3] = width - x1
        return data


@registry.register_processor("detr_processor")
class DetrProcessor(BaseProcessor):
    """
    To encode image with resnet grid features, which is much faster than region feature.
    Detr_processor uses high image resolutions, as pointed out by paper[1], two key factors
    are important for the success of grid feature:
     - the high spatial resolution of the input images
     - object bboxes and attribute annotations
    As for the feature format itself – region or grid – it only affects accuracy minimally.

    DetrProcessor is widely used in following works:
    [1]. In Defense of Grid Features for Visual Question Answering
    [2]. Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers
    [3]. E2E-VLP: End-to-End Vision-Language Pre-training Enhanced by Visual Learning
    [4]. Detr: End-to-End Object Detection with Transformers
    """

    @dataclass
    class Config(AntMMFConfig):
        scales: List[int] = None
        max_size: int = 1333
        num_box_max: Optional[int] = None  # padding to max number of box
        pad_value: Optional[int] = 0  # padding value should be num_categries+1

    def __init__(
        self, config: Union[Configuration, Config, Dict] = None, *args, **kwargs
    ):
        super().__init__(config, *args, **kwargs)

        if self.config.scales is None:
            # default training scales from detr
            self.config.scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if not isinstance(config.scales, collections.abc.Sequence):
            self.config.scales = [self.config.scales]

        normalize = T.DetectionCompose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform = T.DetectionCompose(
            [
                T.RandomResize(self.config.scales, max_size=self.config.max_size),
                normalize,
            ]
        )

    def preprocess_target(self, image, target):
        """
        :param target:
        :return:
        """
        assert check_required_keys(target, ["bbox", "objects"])
        w, h = image.size
        boxes = target["bbox"]  # N,4
        classes = target["objects"]  # N

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = torch.tensor(classes, dtype=torch.int64)

        # filter unreasonable bbox
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        coco_target = {}
        coco_target["keep"] = keep
        coco_target["boxes"] = boxes
        coco_target["labels"] = classes
        coco_target["orig_size"] = torch.as_tensor([int(h), int(w)])
        coco_target["size"] = torch.as_tensor([int(h), int(w)])
        return coco_target

    def pad_target(self, target, num_box_max=None):
        num_box = len(target["boxes"])
        num_box_max = num_box_max or self.config.num_box_max
        assert num_box_max is not None and num_box <= num_box_max

        dtype, device = target["boxes"].dtype, target["boxes"].device
        _boxes = target["boxes"]
        target["boxes"] = torch.zeros([num_box_max, 4], dtype=dtype, device=device)
        target["boxes"][:num_box, :].copy_(_boxes)

        dtype, device = target["labels"].dtype, target["labels"].device
        _labels = target["labels"]
        target["labels"] = torch.zeros([num_box_max], dtype=dtype, device=device).fill_(
            self.config.pad_value
        )
        target["labels"][:num_box].copy_(_labels)

        target["num_box"] = torch.tensor(num_box, dtype=dtype, device=device)

        return target

    def __call__(self, data):
        """
        Args:
            data: {'image': image_arr, # pil RGB format,
                   'target': None or {'bbox': annotations, # [x1,y1,x2,y2], Nx4
                                      'objects' # N,}
                   }
        Returns:
        """

        if isinstance(data, (Image.Image,)):
            return self.transform(data, None)[0]
        target = self.preprocess_target(data["image"], data["target"])
        trans_image, trans_target = self.transform(data["image"], target)
        if self.config.num_box_max is not None:
            trans_target = self.pad_target(trans_target)
        result = {"image": trans_image, "target": trans_target}
        return result
