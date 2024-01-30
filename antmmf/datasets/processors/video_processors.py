# Copyright (c) 2023 Ant Group and its affiliates.
"""
The processors exist in antmmf to make data processing pipelines for video
"""
import random
from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
import torch

from antmmf.common.constants import VISION_MODALITY
from antmmf.common.registry import registry
from antmmf.datasets.processors.processors import BaseProcessor
from antmmf.common import Configuration


@registry.register_processor("fmpeg")
class FMpegProcessor(BaseProcessor):
    """Use fmpeg for video extraction

    Args:
        config (Configuration): node containing configuration parameters of
                             the processor
    """

    @dataclass
    class Config:
        num_frames: Optional[int] = None
        num_sec: Optional[float] = None
        num_clip: int = 1
        size: int = 224
        fps: int = 16
        crop_only: bool = False
        center_crop: bool = True
        keep_ratio: bool = True
        normalization_factor: float = 255.0
        flip: bool = False

    def __init__(
        self, config: Union[Configuration, Config] = Config(), *args, **kwargs
    ):
        if isinstance(config, Configuration):
            self.config = FMpegProcessor.Config(**config)
        else:
            self.config = config
        # if not indicated, return total frames as self._get_duration(video)*self.fps
        self.num_frames = config.num_frames
        self.fps = config.fps  # frame per second
        # num_sec default as whole video duration if not indicated
        self.num_sec = self.num_frames / float(self.fps) if self.num_frames else None
        self.num_clip = config.num_clip

        self.size = (
            config.size
        )  # image size, e.g., if set to 244, the actual image frame is 244 x 244
        self.crop_only = config.crop_only
        self.center_crop = config.center_crop
        self.keep_ratio = config.keep_ratio
        self.normalization_factor = float(config.normalization_factor)
        self.flip = config.flip
        assert isinstance(self.size, int)

    def _get_video(self, video_path, start, end, num_clip):
        if self.num_frames is None:
            # use adaptive padding to get all frames, this will result
            # in various length for video features
            video = self._get_video_start(video_path, 0, end - start, num_frames=None)
            return video.unsqueeze(0)  # num_clip, C, T, H, W
        else:
            # truncate/padding to num_frames
            video = torch.zeros(
                num_clip, 3, self.num_frames, self.size, self.size, dtype=torch.float32
            )
            start_ind = np.linspace(
                start, max(start, end - self.num_sec - 0.4), num_clip
            )
            for i, s in enumerate(start_ind):
                # ensure each clip's feature length is self.num_frames
                video[i] = self._get_video_start(
                    video_path, s, self.num_sec, self.num_frames
                )
            return video  # num_clip, C, T, H, W

    def _get_video_dim(self, video_path):
        import ffmpeg as fp

        probe = fp.probe(video_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        return height, width

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def _get_video_start(self, video_path, start, duration, num_frames=None):
        """

        Args:
            video_path(str): video path.
            start(float): start timestamp of video for sampling frames
            duration(float): sampling duration of video
            num_frames(int): padding or truncate to num_frames. if not set, will adaptively
                             padding to multiple of self.fps.

        Returns:

        """
        import ffmpeg as fp

        start_seek = start
        # see ffmpeg param settings at: https://ffmpeg.org/ffmpeg.html#Main-options
        cmd = fp.input(video_path, ss=start_seek, t=duration + 0.1).filter(
            "fps", fps=self.fps
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = cmd.crop(
                "(iw - {})*{}".format(self.size, aw),
                "(ih - {})*{}".format(self.size, ah),
                str(self.size),
                str(self.size),
            )
        elif self.keep_ratio:  # crop and resize
            # keep ratio crop
            h, w = self._get_video_dim(video_path)
            height, width = self._get_output_dim(h, w)
            cmd = cmd.filter("scale", width, height)

            x = int((width - self.size) * aw)
            y = int((height - self.size) * ah)
            cmd = cmd.crop(x, y, self.size, self.size)
        else:  # crop without keeping ratio
            cmd = cmd.crop(
                "(iw - min(iw,ih))*{}".format(aw),
                "(ih - min(iw,ih))*{}".format(ah),
                "min(iw,ih)",
                "min(iw,ih)",
            ).filter("scale", self.size, self.size)
        out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
            capture_stdout=True, quiet=True
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = torch.from_numpy(video.astype("float32"))  # checkpoint finish
        video = video.permute(3, 0, 1, 2)  # C x T x H x W

        # normalization
        video = video / self.normalization_factor

        # 指定self.num_frames, 则按self.num_frames 和 self.fps
        # 从视频中抽取n_segments = math.floor(self.num_frames/self.fps)
        if num_frames and video.shape[1] < num_frames:
            zeros = video.new_zeros(
                3, num_frames - video.shape[1], self.size, self.size
            )
            video = torch.cat((video, zeros), axis=1)
            video = video[:, :num_frames]  # C x num_frames x H x W
        else:  # padding to multiplier of fps: 按self.fps抽取全部视频, n_segments = math.floor(duration * self.fps)

            def _zero_pad_to_multiple(tensor, size):
                n = size - tensor.size(1) % size
                if n == size:
                    return tensor
                else:
                    z = tensor.new_zeros(
                        tensor.shape[0], n, tensor.shape[2], tensor.shape[3]
                    )
                    return torch.cat((tensor, z), 1)

            video = _zero_pad_to_multiple(video, self.fps)  # C x T x H x W
        return video

    def _get_duration(self, video_path):
        import ffmpeg as fp

        probe = fp.probe(video_path)
        return probe["format"]["duration"]

    def __call__(self, item):
        """Call requires item to have either "image" attribute or either
        "feature" attribute. If "image" is present, it will processed using
        an image processor.

        Args:
            item (Dict): Dict containing the
            "image" : the diretory of the video files to be processed

        Returns:
            Dict: Dict containing indices in "VISION" key

        """
        if not isinstance(item, dict):
            raise TypeError(
                "Argument passed to the processor must be a dict with either 'image' or 'feature' as keys"
            )
        video_path = item.get("image")
        assert video_path is not None
        duration = self._get_duration(video_path)
        video = self._get_video(video_path, 0, float(duration), self.num_clip)

        if self.flip:
            video = torch.cat((video, torch.flip(video, [4])), dim=0)

        ret = {VISION_MODALITY: video, "video_path": video_path}
        return ret
