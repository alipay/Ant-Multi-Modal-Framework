# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

"""
Modified based on:
https://github.com/showlab/all-in-one/blob/
9917ef5ada58c36c46f24b6b6ce437235620c308/AllInOne/datasets/video_base_dataset.py
"""

import os
import random

import cv2
import decord
import numpy as np
import torch
from decord import cpu


class VideoReader(object):
    def __init__(self, training=False, num_frm=1):
        self.training = training
        self.num_frm = num_frm

    def sample_frames(
        self,
        num_clips,
        vlen,
        sample="rand",
        num_frame=1,
        fix_start=None,
        frame_resample="uniform",
    ):
        acc_samples = min(num_clips, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []  # 划分clip区间
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        # incase ranges are empty
        ranges = np.int32(ranges)
        idx = np.where(ranges[:, 0] >= ranges[:, 1])[0]
        if len(idx) > 0:  # fix range
            cor_range = ranges[idx]
            cor_range[:, 1] = cor_range[:, 0] + 1
            ranges[idx] = cor_range

        if sample == "rand":  # 每个clip里抽一帧
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        elif fix_start is not None:
            frame_idxs = [x[0] + fix_start for x in ranges]
        elif sample == "uniform":  # 每个clip里抽中间帧
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if acc_samples < num_clips:
            frame_idxs_new = []
            if frame_resample == "uniform":
                for i in range(num_clips):
                    frame_idxs_new.append(
                        frame_idxs[int((acc_samples - 1) * i / (num_clips - 1) + 0.5)]
                    )
                frame_idxs = frame_idxs_new
            else:
                repeat_nums = (num_clips - 1) // acc_samples + 1
                for i in range(acc_samples):
                    frame_idxs_new += [frame_idxs[i]] * repeat_nums
                if frame_resample == "front":
                    frame_idxs = frame_idxs_new[:num_clips]
                elif frame_resample == "back":
                    frame_idxs = frame_idxs_new[-num_clips:]
                else:
                    raise NotImplementedError

        return frame_idxs

    def read_frames_decord(
        self, video_path, num_clips, fix_start=None, begin_time=None, end_time=None
    ):
        # print("video path: {}".format(video_path))
        if self.training:
            sample = "rand"
        else:
            sample = "uniform"
        video_reader = decord.VideoReader(
            video_path, width=-1, height=-1, num_threads=0, ctx=cpu(0)
        )
        # video_reader = decord.VideoReader(video_path, width=256, height=256, num_threads=1, ctx=cpu(0))
        decord.bridge.set_bridge("torch")
        vlen = len(video_reader)
        if begin_time is not None or end_time is not None:
            average_fps = video_reader.get_avg_fps()
            clip_len = (end_time - begin_time) * average_fps
            frame_idxs = self.sample_frames(num_clips, int(clip_len), sample=sample)
            rel_index = int(begin_time * average_fps)
            rel_index = max(rel_index, 0)
            rel_index = min(rel_index, vlen - 1)
            frame_idxs_abs = (
                (np.int32(frame_idxs) + np.int32(rel_index)).clip(0, vlen - 1).tolist()
            )
            frames = video_reader.get_batch(frame_idxs_abs).byte()
            fs = video_reader.get_frame_timestamp(frame_idxs_abs)
            interval = [fs.min(), fs.max()]
        else:
            frame_idxs = self.sample_frames(
                num_clips, vlen, sample=sample, num_frame=1, fix_start=fix_start
            )
            frames = video_reader.get_batch(frame_idxs).byte()
            fs = video_reader.get_frame_timestamp(frame_idxs)
            interval = [fs.min(), fs.max()]
        frames = frames.permute(0, 3, 1, 2)  # n, c, h, w

        # import matplotlib.pylab as plt
        # frames = frames.permute(0, 2, 3, 1)/255.
        # for i in range(1, num_clips+1):
        #     plt.subplot(1, num_clips, i)
        #     plt.imshow(frames[i-1])
        # plt.show()
        return frames, frame_idxs, vlen

    def read_frames_from_img_dir(
        self, video_path, num_frames, fix_start=None, frame_resample="uniform"
    ):
        if self.training:
            sample = "rand"
        else:
            sample = "uniform"
        framelist = os.listdir(video_path)
        vlen = len(framelist)
        frame_idxs = self.sample_frames(
            num_frames,
            vlen,
            sample=sample,
            fix_start=fix_start,
            frame_resample=frame_resample,
        )
        sortlist = []
        for imgdir in framelist:
            frameid = imgdir.split("/")[-1]
            frameid = frameid.split(".")[0]
            frameid = frameid.split("_")[-1]
            frameid = frameid.replace("frame", "")
            sortlist.append([int(frameid), imgdir])
        sortlist.sort()
        frames = []
        for idx in frame_idxs:
            frame_name = sortlist[idx][1]
            frame = cv2.imread(os.path.join(video_path, frame_name))
            frame = torch.from_numpy(frame).byte()
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
        frames = torch.stack(frames, 0)  # n, c, h, w
        return frames, frame_idxs, vlen
