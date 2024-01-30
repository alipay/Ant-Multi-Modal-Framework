# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from dataclasses import dataclass
from typing import Union, Optional, Tuple
from antmmf.common.registry import registry
from antmmf.datasets.processors.processors import BaseProcessor
from antmmf.common import Configuration, AntMMFConfig


@registry.register_processor("pyvideo_transform")
class VideoProcessor(BaseProcessor):
    """Processor for video transform
    wrapper for import pytorchvideo.transforms.create_video_transform
    see doc:
    https://pytorchvideo.readthedocs.io/en/latest/api/transforms/transforms.html

    Args:
        config (Configuration): node containing configuration parameters of
                             the processor
    """

    @dataclass
    class Config(AntMMFConfig):
        mode: str = "train"
        video_key: Optional[str] = None
        num_samples: Optional[int] = None
        video_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
        video_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
        crop_size: Union[int, Tuple[int, int]] = 224
        horizontal_flip_prob: float = 0.5
        aug_type: str = "default"

    def __init__(
        self, config: Union[Configuration, Config] = Config(), *args, **kwargs
    ):
        self.config = self.__class__.Config.create_from(config, **kwargs)
        import pytorchvideo.transforms as video_transforms

        self.transform = video_transforms.create_video_transform(
            mode=self.config.mode,
            min_size=int(self.config.crop_size * 1.2),
            max_size=int(self.config.crop_size * 1.5),
            crop_size=self.config.crop_size,
            video_std=self.config.video_std,
            video_mean=self.config.video_mean,
            aug_type=self.config.aug_type,  # randaug/augmix
            num_samples=self.config.num_samples,  # not use temporal sub sampling
        )

    def __call__(self, video, video_format="TCHW"):
        if self.config.video_key is not None:
            video = video[self.config.video_key]
        if video_format == "TCHW":
            video = video.transpose(0, 1)
        video = self.transform(video)  # format： (C, T, H, W)
        if video_format == "TCHW":
            video = video.transpose(0, 1)
        return video
