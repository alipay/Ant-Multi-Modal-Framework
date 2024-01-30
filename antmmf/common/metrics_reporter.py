# Copyright (c) 2023 Ant Group and its affiliates.
import json
import logging
import multiprocessing as mp
import os
import queue
import statistics
from datetime import datetime

import requests
import torch

logger = logging.Logger(name=__name__)


class MetricsReporter(object):
    def __init__(
        self,
        model,
        model_name,
        batch_data,
        drop_last,
        batch_size,
        model_path="",
        model_type="offline",
        framework="pytorch",
        start_time=datetime.now(),
        rank=0,
        world_size=1,
        num_cache_steps=1,
        **kwargs,
    ):
        self.user_id = os.environ.get("USERNUMBER", "-1")
        self.job_id = self.get_jobid()
        self.model_type = model_type
        self.model_name = model_name
        self.model_path = model_path
        self.start_time = start_time
        self.framework = framework
        self.model = model
        self.batch_data = batch_data
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.global_step = 0  # start from 0
        self.headers_json = {
            "Content-Type": "application/json",
            "postman-token": "7267554f-0922-c93a-0b89-8e95e166a8da",
            "cache-control": "no-cache",
        }
        self.url = "xxx"
        self.num_params = self.get_total_params()
        self.duration = 0
        self.rank = rank
        self.world_size = world_size
        self.aprofiler = None

        self.fwd_flops_per_step_per_gpu = 0
        self.flops_per_step_per_gpu = 0
        self.flops_per_step = 0
        self.reporting_data = {
            "table": "green_ai_alps_train_flops_data",
            "datas": {
                "headers": [
                    "period",
                    "userId",
                    "modelName",
                    "jobId",
                    "modelPath",
                    "dropRemainder",
                    "batchSize",
                    "steps",
                    "trainFLOPs",
                    "startTime",
                    "params",
                    "modelType",
                    "framework",
                    "duration",
                ],
                "datas": [[]],
            },
            "appendOrRecover": 0,
        }
        self.num_cache_steps = num_cache_steps
        self.cached_metrics = []
        self.report_cached_metrics = False
        self.post_error = False
        self.profile_error = False
        self.kwargs = kwargs
        self.queue = None
        self.post_process = None
        self.init_posting_util()

    def __del__(self):
        if isinstance(self.post_process, mp.Process) and self.post_process.is_alive():
            self.post_process.terminate()

    def init_posting_util(self):
        if self.rank == 0:
            self.queue = mp.Queue(maxsize=10)
            self.post_process = mp.Process(
                target=self.post_metrics, args=(self.queue,), daemon=True
            )
            self.post_process.start()

    def get_jobid(self):
        # in antdocker, use "AISTUDIO_JOB_NAME" as jobid
        jobid = os.environ.get("AISTUDIO_JOB_NAME", None)
        if jobid is None:
            # in k8s, use "APP_ID"
            jobid = os.environ.get("APP_ID", None)
        # local running
        if jobid is None:
            jobid = "LOCAL_JOB"
        return jobid

    def no_need_profile(self):
        return (
            self.aprofiler is None
            or self.job_id
            == "LOCAL_JOB"  # CUDA not available  # train at local machine, not on the cluster
        )

    def disable_profiler(self):
        return (
            self.no_need_profile()
            or self.report_cached_metrics is True
            or self.profile_error is True
        )

    def not_report(self):
        return (
            self.no_need_profile()
            or self.rank > 0
            or self.post_error is True
            or self.profile_error is True
        )

    def dryrun(self):
        try:
            self.model(self.batch_data)
        except Exception:
            self.profile_error = True
        if self.profile_error:
            self.__del__()
            self.aprofiler.end_profile()
        else:
            if self.rank > 0:
                self.aprofiler.end_profile()
            else:
                self.aprofiler._reset_profiler()

    # 给model挂上用于计算FLOPs的hook
    def start_profile(self, global_step=None):
        if self.no_need_profile():
            return
        self.aprofiler.start_profile()
        if self.global_step == 0:
            self.dryrun()
        if global_step is not None:
            self.update_global_step(global_step)

    # 将aprofiler记录的FLOPs等信息归零
    def reset_profile(self, global_step=None):
        if self.disable_profiler():
            return
        self.aprofiler._reset_profiler()
        if global_step is not None:
            self.update_global_step(global_step)

    # 停止profile，删掉start_profile挂上的所有hook
    def end_profile(self):
        if self.no_need_profile():
            return
        self.aprofiler.end_profile()

    # 模型参数数量
    def get_total_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_total_fwd_flops_per_gpu(self):
        if self.disable_profiler():
            return 0
        # 单个GPU上，前向计算的FLOPs
        try:
            flops = self.aprofiler.get_total_flops()
        except Exception:
            self.profile_error = True
            flops = 0
        return flops

    def get_train_flops(self):
        # 所有GPU前向计算+反向传播的FLOPs
        if self.report_cached_metrics:
            return statistics.mean(self.cached_metrics)
        # 反向FLOPs约为前向的2倍
        return self.get_total_fwd_flops_per_gpu() * self.world_size * 3

    def update_global_step(self, global_step):
        self.global_step = global_step

    def report(self, train_flops, global_step=None):
        # train_flops需要用户来传递。
        if self.not_report():
            return
        if self.num_cache_steps > 0 and len(self.cached_metrics) < self.num_cache_steps:
            self.cached_metrics.append(train_flops)
        elif (
            self.num_cache_steps > 0
            and len(self.cached_metrics) == self.num_cache_steps
        ):
            self.end_profile()
            self.report_cached_metrics = True

        global_step_to_report = (
            global_step if global_step is not None else self.global_step
        )
        curr_dt = datetime.now()
        duration = (curr_dt - self.start_time).total_seconds()
        timestamp = int(curr_dt.timestamp() * 1000)
        current_step_reporting_data = [
            timestamp,  # period
            self.user_id,  # userId
            f"antmmf-{self.model_name}",  # modelName
            self.job_id,  # jobId
            self.model_path,  # modelPath
            self.drop_last,  # dropRemainder
            float(self.batch_size * self.world_size),  # batchSize
            float(global_step_to_report),  # steps
            float(train_flops),  # trainFLOPs
            datetime.strftime(self.start_time, "%Y-%m-%d %H:%M:%S"),  # startTime
            float(self.num_params),  # params
            self.model_type,  # modelType
            self.framework,  # framework
            duration,  # duration
        ]

        self.reporting_data["datas"]["datas"][0] = current_step_reporting_data
        # put数据到长度为10的队列里，如果队列满了，再等1秒，
        # 还不能put则视为上报出错，之后不再上报
        try:
            self.queue.put(
                [self.url, self.reporting_data, self.headers_json], timeout=1
            )
        except queue.Full:
            self.post_error = True

    @staticmethod
    def post_metrics(queue):
        while True:
            elem = queue.get()
            url, data, headers = elem
            try:
                # url, data, json见上报样例
                r = requests.post(
                    url,
                    data=json.dumps(data),
                    headers=headers,
                )
                if r.status_code != requests.codes.ok:
                    return
            except Exception:
                return
