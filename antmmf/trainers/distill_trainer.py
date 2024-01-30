# Copyright (c) 2023 Ant Group and its affiliates.
import math
from itertools import chain

import torch
from antmmf.common import constants
from antmmf.common.registry import registry
from antmmf.common.report import Report
from antmmf.common.task_loader import TaskLoader
from antmmf.trainers.base_trainer import BaseTrainer, check_configuration
from antmmf.models.build import build_pretrained_model
from antmmf.utils.distributed_utils import reduce_dict
from antmmf.utils.general import nullcontext
from antmmf.utils.logger import Logger
from antmmf.utils.timer import Timer
from tqdm import tqdm


@registry.register_trainer("distill_trainer")
class DistillTrainer(BaseTrainer):
    ALLOWED_RUNTYPE = [
        "train",
        "train+val",
    ]

    def __init__(self, config):
        super().__init__(config)

    def load(self, has_check_point=True):
        self._init_process_group()
        self.writer = Logger(self.config)

        self.run_type = self.config.training_parameters.get("run_type", "train")
        assert (
            self.run_type in DistillTrainer.ALLOWED_RUNTYPE
        ), "unrecognized run_type:{}".format(self.run_type)

        # flag indicates train/val/test state, not online serving
        registry.register(constants.STATE, constants.STATE_LOCAL)

        self.task_loader = TaskLoader(self.config)

        self.configuration = check_configuration(registry.get("configuration"))
        self.writer.write(self.configuration)

        self.config_based_setup()

        self.load_task()
        self.load_model()
        self.load_optimizer()
        self.load_extras(has_check_point)
        self._build_loader_list()

    def _load_teacher_model(self, model_key, attributes):
        attributes["model"] = model_key
        self.teacher, model_config = build_pretrained_model(attributes)

    def load_model(self):
        tp = self.config.training_parameters
        assert (
            tp.teacher in self.config.model_attributes
        ), "Teacher model should be in model_attributes"
        assert (
            tp.student in self.config.model_attributes
        ), "Student model should be in model_attributes"

        attributes = self.config.model_attributes[tp.student]
        self._load_model(attributes.model_class, attributes)
        attributes = self.config.model_attributes[tp.teacher]
        self._load_teacher_model(tp.teacher, attributes)
        data_parallel = tp.data_parallel
        distributed = tp.distributed

        registry.register("data_parallel", data_parallel)
        registry.register("distributed", distributed)

        if "cuda" in str(tp.device):
            rank = self.local_rank if self.local_rank is not None else 0
            device_info = "CUDA Device {} is: {}".format(
                rank, torch.cuda.get_device_name(self.local_rank)
            )

            self.writer.write(device_info, log_all=True)

        if data_parallel:
            self.writer.write(
                "cuda device_list:{}  main device:{}".format(
                    self.device_list, self.device
                )
            )

        # pytorch>=1.5 does not support device transfer grammar:
        # tensor.to('device:0,1,2')
        self.model = self.model.to(self.device)
        self.teacher = self.teacher.to(self.device)

        self.writer.write("Torch version is: " + torch.__version__)

        if (
            "cuda" in str(self.device)
            and torch.cuda.device_count() > 1
            and data_parallel is True
        ):
            self.writer.write("Using DataParallel")
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_list)
            self.teacher = torch.nn.DataParallel(
                self.teacher, device_ids=self.device_list
            )

        if (
            "cuda" in str(self.device)
            and self.local_rank is not None
            and distributed is True
        ):
            self.writer.write("Using DistributedDataParallel")

            # SyncBatchNorm only support DistributedDataParallel mode
            if tp.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.teacher
                )
                self.writer.write("Using SyncBatchNorm")

            torch.cuda.set_device(self.local_rank)
            # set find_unused_parameters=True, see issue:
            # https://github.com/open-mmlab/mmdetection/issues/2153
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], find_unused_parameters=True
            )
            self.teacher = torch.nn.parallel.DistributedDataParallel(
                self.teacher, device_ids=[self.local_rank], find_unused_parameters=True
            )

    def train(self):
        self.writer.write("===== Model =====")
        self.writer.write(self.model)
        self.writer.write("===== Teacher =====")
        self.writer.write(self.teacher)

        should_break = False

        if self.max_epochs is None:
            self.max_epochs = math.inf
        else:
            self.max_iterations = min(
                self.max_iterations, self.max_epochs * self.epoch_iterations
            )

        self.teacher.eval()
        self.model.train()
        self.train_timer = Timer()
        self.snapshot_timer = Timer()

        self.profile("Setup Time")

        # This mode should be enabled only for debugging as the different tests
        # will slow down your program execution. More details reference to:
        # https://pytorch.org/docs/stable/autograd.html?highlight=detect_anomaly#torch.autograd.detect_anomaly
        # torch.autograd.set_detect_anomaly(True)

        self.enable_amp = (
            "cuda" in str(self.device)
            and self.training_parameters.enable_amp
            and torch.__version__ >= "1.7.0"
        )
        if self.enable_amp:
            self.writer.write("Using Automatic mixed precision training")
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)

        self.writer.write("Start distilling...")
        while self.current_iteration < self.max_iterations and not should_break:
            self.current_epoch += 1
            registry.register("current_epoch", self.current_epoch)

            # Seed the sampler in case if it is distributed
            self.task_loader.seed_sampler("train", self.current_epoch)

            if self.current_epoch > self.max_epochs:
                break

            for batch in tqdm(
                chain(*self.train_loader_list),
                total=self._len_of_loader_list(self.train_loader_list),
            ):
                self.profile("Batch load time")
                self.current_iteration += 1
                self.writer.write(self.current_iteration, "debug")
                registry.register("current_iteration", self.current_iteration)
                if self.current_iteration > self.max_iterations:
                    break
                report, _, _ = self._forward_pass(batch, enable_amp=self.enable_amp)

                self._update_meter(report, self.meter)

                loss = self._extract_loss(report)
                self._backward(loss)
                should_break = self._logistics()

                self._run_scheduler()

                if should_break:
                    break

        self.finalize()

    def _forward_pass(self, batch, enable_amp=False):
        prepared_batch = self.task_loader.prepare_batch(batch)
        if prepared_batch is None:
            return None, None, None

        self.profile("Batch prepare time")
        forward_context = (
            torch.cuda.amp.autocast(enabled=True) if enable_amp else nullcontext()
        )

        with forward_context:
            teacher_output = self.teacher(prepared_batch)
            model_output = self.model(prepared_batch, teacher_output=teacher_output)

        report = Report(
            prepared_batch,
            model_output,
            {
                "teacher_losses": teacher_output["losses"],
                "teacher_metrics": teacher_output["metrics"],
            },
        )

        self.profile("Forward time")
        return report, model_output, prepared_batch

    def _update_meter(self, report, meter):
        losses_dict = report.losses
        metrics_dict = report.metrics
        teacher_losses_dict = {}
        for k, v in report.teacher_losses.items():
            teacher_losses_dict["teacher/" + k] = v
        teacher_metrics_dict = {}
        for k, v in report.teacher_metrics.items():
            teacher_metrics_dict["teacher/" + k] = v

        reduced_loss_dict = reduce_dict(losses_dict)
        reduced_metrics_dict = reduce_dict(metrics_dict)
        reduced_teacher_loss_dict = reduce_dict(teacher_losses_dict)
        reduced_teacher_metrics_dict = reduce_dict(teacher_metrics_dict)

        with torch.no_grad():
            meter_update_dict = {}
            meter_update_dict.update(reduced_loss_dict)
            meter_update_dict.update(reduced_metrics_dict)
            meter_update_dict.update(reduced_teacher_loss_dict)
            meter_update_dict.update(reduced_teacher_metrics_dict)
            meter.update(meter_update_dict)
