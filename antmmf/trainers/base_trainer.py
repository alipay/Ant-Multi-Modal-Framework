# Copyright (c) 2023 Ant Group and its affiliates.
import copy
import gc
import math
import os
import time
import warnings
from distutils.version import LooseVersion
from itertools import chain

import torch
from torch import nn, optim
from tqdm import tqdm

from antmmf.common import constants
from antmmf.common.checkpoint import Checkpoint
from antmmf.common.configuration import pretty_print, Configuration, string_to_bool
from antmmf.common.meter import Meter
from antmmf.common.metrics_reporter import MetricsReporter
from antmmf.common.registry import registry
from antmmf.common.report import Report
from antmmf.common.task_loader import TaskLoader
from antmmf.models.build import build_model
from antmmf.models.build import build_pretrained_model
from antmmf.modules.build import build_interpreter
from antmmf.modules.metrics import Metrics
from antmmf.optimizer.build import build_optimizer
from antmmf.optimizer.combine_optimizers import CombinedOptimizer
from antmmf.utils.distributed_utils import (
    broadcast_scalar,
    is_main_process,
    reduce_dict,
    synchronize,
    get_rank,
    get_world_size,
)
from antmmf.utils.early_stopping import EarlyStopping
from antmmf.utils.env import set_seed
from antmmf.utils.general import clip_gradients, lr_lambda_update, nullcontext
from antmmf.utils.general import count_parameters
from antmmf.utils.general import get_package_version
from antmmf.utils.logger import Logger
from antmmf.utils.register_fp32 import set_escapes_class_fp32
from antmmf.utils.timer import Timer


def check_configuration(config: Configuration):
    """
    Check the whether the configuration is valid, if not, we will edit it/them.

    Args:
        config (Configuration): The configuration must include the default configuration
        show in antmmt/common/defaults/configs/base.yml
    """
    config.defrost()
    tp = config["training_parameters"]
    if tp["seed"] is not None:
        if is_main_process():
            print(
                "You have chosen to seed the training. This will turn on CUDNN deterministic "
                "setting which can slow down your training considerably! You may see unexpected "
                "behavior when restarting from checkpoints."
            )
    if (
        not torch.cuda.is_available()
        and "cuda" in config["training_parameters"]["device"]
    ):
        if is_main_process():
            print(
                "WARNING: Device specified is 'cuda' but cuda is not present. Switching to CPU version"
            )
        config["training_parameters"]["device"] = "cpu"
    if tp["distributed"] is True and tp["data_parallel"] is True:
        if is_main_process():
            print(
                "training_parameters.distributed and "
                "training_parameters.data_parallel are "
                "mutually exclusive. Setting "
                "training_parameters.distributed to False"
            )
        tp["distributed"] = False

    config.training_parameters.distributed = string_to_bool(
        config.training_parameters.distributed
    )
    config.freeze()
    return config


@registry.register_trainer("base_trainer")
class BaseTrainer:
    ALLOWED_RUNTYPE = [
        "train",
        "val",
        "inference",
        "predict",
        "interpret",
        "train+val",
        "train+inference",
        "train+predict",
    ]

    def __init__(self, config):
        self.config = check_configuration(config)
        self.profiler = Timer()
        self.first_val_batch = None
        self.writer = None
        self.run_type = None
        self.task_loader = None
        self.disable_tqdm = False

    def load(self, has_check_point=True):
        # A100 have issue on tf32
        # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        # This flag defaults to True in PyTorch 1.7 to PyTorch 1.11, and False in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        self._init_process_group()
        self.writer = Logger(self.config)

        self.run_type = self.config.training_parameters.get("run_type", "train")
        self.disable_tqdm = self.config.training_parameters.disable_tqdm
        assert (
            self.run_type in BaseTrainer.ALLOWED_RUNTYPE
        ), "unrecognized run_type:{}".format(self.run_type)

        # flag indicates train/val/test state, not online serving
        registry.register(constants.STATE, constants.STATE_LOCAL)
        registry.register("writer", self.writer)

        self.task_loader = TaskLoader(self.config)

        pretty_print(self.config)

        self.config_based_setup()

        self.load_task()
        self.load_model()
        self.load_extras(has_check_point)
        self._build_loader_list()

    def _build_loader_list(self):
        def build_order(name_order, load_mapping):
            loader_order = []
            for name in name_order:
                loader_order.append(load_mapping[name])
            return loader_order

        for t_type in self.task_loader.task_type:
            if t_type == "train":
                self.train_loader_list = build_order(
                    self.dataset_train_order, self.train_loader
                )
            if t_type == "val":
                self.val_loader_list = build_order(
                    self.dataset_val_order, self.val_loader
                )
            if t_type == "test":
                self.test_loader_list = build_order(
                    self.dataset_test_order, self.test_loader
                )
            if t_type == "interpret":
                self.interpret_loader_list = build_order(
                    self.dataset_interpret_order, self.interpret_loader
                )

    def _len_of_loader_list(self, loader_list):
        length = 0
        for loader in loader_list:
            length += len(loader)
        return length

    def _cuda_device_list(self):
        device_list = str(self.device).split(":") if "cuda" in self.device else None
        if device_list is not None:
            if len(device_list) == 1:
                device_list = [0]
            else:
                assert (
                    len(device_list) == 2
                ), r'has to be the format of "cuda:<id0>,<id1>,...,<idn>", e.g., cuda:1,0'
                device_list = [int(d) for d in device_list[-1].split(",")]
        return device_list

    def _init_process_group(self):
        training_parameters = self.config.training_parameters
        self.device = training_parameters.device
        self.device_list = self._cuda_device_list()
        if training_parameters.distributed:
            if not torch.distributed.is_nccl_available():
                raise RuntimeError(
                    "Unable to initialize process group: NCCL is not available"
                )

            self.local_rank = training_parameters.local_rank
            if self.local_rank is not None:
                torch.distributed.init_process_group(backend="nccl")
                synchronize()
        else:
            self.local_rank = None

        # decide current device
        # for distributed dp
        if (
            "cuda" in self.device
            and training_parameters.distributed
            and self.local_rank is not None
        ):
            self.device = torch.device("cuda", self.local_rank)

        # for data_parallel
        if (
            "cuda" in str(self.device)
            and torch.cuda.device_count() > 1
            and training_parameters.data_parallel is True
        ):
            self.device = "cuda:{}".format(self.device_list[0])

        registry.register("current_device", self.device)

    def load_task(self):
        self.writer.write("Loading tasks and data", "info")
        self.task_loader.load_task()

        self.task_loader.make_dataloaders()

        # delegate loader & task to trainer for easy access
        for t_type in self.task_loader.task_type:
            if t_type == "train":
                self.train_task = self.task_loader.task_mapping["train"]
                self.train_loader = self.task_loader.loader_mapping["train"]
                self.epoch_iterations = (
                    len(self.train_task) // self.config.training_parameters.batch_size
                    + 1
                )
            elif t_type == "val":
                self.val_task = self.task_loader.task_mapping["val"]
                self.val_loader = self.task_loader.loader_mapping["val"]
                # Total iterations for snapshot when evaluation
                self.snapshot_iterations = len(self.val_task)
                self.snapshot_iterations //= self.config.training_parameters.batch_size
            elif t_type == "interpret":
                self.interpret_task = self.task_loader.task_mapping["interpret"]
                self.interpret_loader = self.task_loader.loader_mapping["interpret"]
            else:  # must be test task
                self.test_task = self.task_loader.task_mapping["test"]
                self.test_loader = self.task_loader.loader_mapping["test"]

    def _load_model(self, model_key, attributes):
        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_attributes[attributes]

        attributes.defrost()
        attributes["model"] = model_key
        attributes.freeze()

        self.task_loader.update_registry_for_model(attributes)
        self.model = build_model(attributes)
        self.model_name = attributes.model
        self.task_loader.clean_config(attributes)
        self.overall_metric_evaluator = Metrics(attributes.get("metrics", []))

    def _load_pretrained_model(self, model_key, attributes):
        attributes.defrost()
        attributes["model"] = model_key
        attributes.freeze()

        self.model, model_config = build_pretrained_model(attributes)
        self.model_name = attributes.from_pretrained.model_name
        self.task_loader.update_registry_for_model(model_config)
        self.task_loader.clean_config(model_config)
        self.overall_metric_evaluator = Metrics(model_config.get("metrics", []))

    def load_model(self):
        assert (
            len(self.config.model_attributes) == 1
        ), "There should be only one model in model_attributes"
        model_key = list(self.config.model_attributes.keys())[0]
        attributes = self.config.model_attributes[model_key]
        if "from_pretrained" in attributes:
            self._load_pretrained_model(model_key, attributes)
        else:
            self._load_model(model_key, attributes)

        if (
            hasattr(self.config, "amp_attributes")
            and self.config.training_parameters.enable_amp
        ):
            set_escapes_class_fp32(self.model, self.config.amp_attributes)

        training_parameters = self.config.training_parameters
        data_parallel = training_parameters.data_parallel
        distributed = training_parameters.distributed
        # let ddp knows if computation graph is static will potentially improve performance
        # and enable gradient_checkpoint:
        # refer to: https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel
        static_graph = training_parameters.static_graph

        registry.register("data_parallel", data_parallel)
        registry.register("distributed", distributed)

        if "cuda" in str(self.config.training_parameters.device):
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
        self.load_optimizer()

        self.writer.write("Torch version is: " + torch.__version__)

        if (
            "cuda" in str(self.device)
            and torch.cuda.device_count() > 1
            and data_parallel is True
        ):
            self.writer.write("Using DataParallel")
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_list)

        # set auto mixed precision training configuration
        self.enable_amp = (
            "cuda" in str(self.device)
            and self.config.training_parameters.enable_amp
            and torch.__version__ >= "1.7.0"
        )
        self.enable_torch_amp = False
        if self.enable_amp:
            amp_opt_level = "O1"
            if hasattr(self.config, "amp_attributes") and hasattr(
                self.config.amp_attributes, "opt_level"
            ):
                amp_opt_level = self.config.amp_attributes.opt_level
            if amp_opt_level.upper() not in ("O1", "O2", "O3"):
                raise ValueError('amp opt_level only support "O1", "O2" or "O3"')
            if amp_opt_level == "O1":
                self.enable_torch_amp = True

        if (
            "cuda" in str(self.device)
            and distributed is True
            and self.local_rank is not None
        ):
            self.writer.write("Using DistributedDataParallel")

            # SyncBatchNorm only support DistributedDataParallel mode
            if self.config.training_parameters.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.writer.write("Using SyncBatchNorm")

            torch.cuda.set_device(self.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.config.training_parameters.find_unused_parameters,
            )
            if static_graph and LooseVersion(get_package_version("torch")) >= "1.9.0":
                # static_graph is support for torch>=1.9
                self.model._set_static_graph()

    def load_optimizer(self):
        self.optimizer = build_optimizer(self.model, self.config)

    def load_extras(self, has_check_point=True):
        self.checkpoint = None if has_check_point is False else Checkpoint(self)
        self.meter = Meter()

        self.training_parameters = self.config.training_parameters

        monitored_metric = self.training_parameters.monitored_metric
        metric_minimize = self.training_parameters.metric_minimize
        should_early_stop = self.training_parameters.should_early_stop
        patience = self.training_parameters.patience

        self.log_interval = self.training_parameters.log_interval
        self.snapshot_interval = self.training_parameters.snapshot_interval
        self.max_iterations = self.training_parameters.max_iterations
        self.should_clip_gradients = self.training_parameters.clip_gradients
        self.max_epochs = self.training_parameters.max_epochs
        self.gradient_accumulation_steps = int(
            self.training_parameters.gradient_accumulation_steps
        )
        assert self.gradient_accumulation_steps >= 1
        for t_type in self.task_loader.task_type:
            if t_type == "train":
                self.dataset_train_order = self.training_parameters.get(
                    "dataset_train_order", self.train_task.datasets_name
                )
            if t_type == "val":
                self.dataset_val_order = self.training_parameters.get(
                    "dataset_val_order", self.val_task.datasets_name
                )
            if t_type == "test":
                self.dataset_test_order = self.training_parameters.get(
                    "dataset_test_order", self.test_task.datasets_name
                )
            if t_type == "interpret":
                self.dataset_interpret_order = self.training_parameters.get(
                    "dataset_interpret_order", self.interpret_task.datasets_name
                )

        self.early_stopping = EarlyStopping(
            self.model,
            self.checkpoint,
            monitored_metric,
            patience=patience,
            minimize=metric_minimize,
            should_stop=should_early_stop,
        )
        self.current_epoch = 1
        self.current_iteration = 0

        if self.checkpoint is not None:
            self.checkpoint.load_state_dict()

        self.not_debug = self.training_parameters.logger_level != "debug"

        self.lr_scheduler = None
        if "train" in self.run_type:
            # only build lr_scheduler for training phase
            self.setup_lr_scheduler()

        # training_parameters.overall_metrics is DEPRECATED.
        # model_attributes.MODEL_NAME.metrics is used as overall metrics.
        # For compatibility reasons, if training_parameters.overall_metrics
        # is set, it will override model_attributes ones.
        if "overall_metrics" in self.training_parameters:
            self.overall_metric_evaluator = Metrics(
                self.config.training_parameters.get("overall_metrics", [])
            )
        self.synchronized_loss = self.config.training_parameters.synchronized_loss

    def setup_lr_scheduler(self):
        # support custom scheduler
        has_custom = hasattr(self.model, "get_custom_scheduler")
        if has_custom:
            self.lr_scheduler = self.model.get_custom_scheduler(self)
        else:
            is_parallel = isinstance(self.model, nn.DataParallel) or isinstance(
                self.model, nn.parallel.DistributedDataParallel
            )
            if is_parallel and hasattr(self.model.module, "get_custom_scheduler"):
                self.lr_scheduler = self.model.module.get_custom_scheduler(self)

        if self.lr_scheduler is None and self.training_parameters.lr_scheduler is True:
            scheduler_class = optim.lr_scheduler.LambdaLR

            def scheduler_func(x):
                return lr_lambda_update(x, self)

            self.lr_scheduler = scheduler_class(
                self.optimizer, lr_lambda=scheduler_func
            )

    def config_based_setup(self):
        seed = self.config.training_parameters.seed
        if seed is None:
            return
        set_seed(seed)

    def train(self):
        self.writer.write("===== Model =====")
        self.writer.write(self.model)
        self.writer.write(
            "Model Params: Trainable {Trainable:.3f}M  Total {Total:.3f}M".format(
                **count_parameters(self.model)
            )
        )

        if "train" not in self.run_type:
            self.inference()
            return

        should_break = False

        if self.max_epochs is None:
            self.max_epochs = math.inf
        else:
            self.max_iterations = min(
                self.max_iterations, self.max_epochs * self.epoch_iterations
            )

        self.model.train()
        self.train_timer = Timer()

        self.profile("Setup Time")

        # This mode should be enabled only for debugging as the different tests
        # will slow down your program execution. More details reference to:
        # https://pytorch.org/docs/stable/autograd.html?highlight=detect_anomaly#torch.autograd.detect_anomaly
        # torch.autograd.set_detect_anomaly(True)

        if self.enable_torch_amp:
            self.writer.write("Using Automatic mixed precision training")
            if hasattr(self.config, "amp_attributes") and hasattr(
                self.config.amp_attributes, "growth_interval"
            ):
                growth_interval = self.config.amp_attributes.growth_interval
            else:
                #  2000 is a default value. More details reference to:
                # https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
                growth_interval = 2000
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=self.enable_torch_amp, growth_interval=growth_interval
            )

        self.optimizer.zero_grad()

        # Fetch the first batch of validation dataloader for MetricReporter and evaluation during training.
        if self.first_val_batch is None:
            self.first_val_batch = next(chain(*self.val_loader_list))

        metrics_reporter = MetricsReporter(
            self.model,
            self.model_name,
            batch_data=copy.deepcopy(self.first_val_batch),
            rank=get_rank(),
            world_size=get_world_size(),
            drop_last=self.train_loader[self.dataset_train_order[0]].drop_last,
            batch_size=self.train_loader[self.dataset_train_order[0]].batch_size,
        )
        metrics_reporter.start_profile()

        self.writer.write("Starting training...")
        while self.current_iteration < self.max_iterations and not should_break:
            registry.register("current_epoch", self.current_epoch)
            # Seed the sampler in case if it is distributed
            self.task_loader.seed_sampler("train", self.current_epoch)

            if self.current_epoch > self.max_epochs:
                break

            for batch in tqdm(
                chain(*self.train_loader_list),
                total=self._len_of_loader_list(self.train_loader_list),
                disable=self.disable_tqdm or (not is_main_process()),
            ):
                self.profile("Batch load time")
                self.current_iteration += 1
                if (
                    self.config.model_attributes.univl.get("hard_example_mining", False)
                    and self.config.model_attributes.univl.get("change_iter", None)
                    is not None
                ):
                    change_iter = self.config.model_attributes.univl.change_iter
                    change_rate = self.config.model_attributes.univl.change_rate
                    thre_num = int(self.current_iteration / change_iter)
                    thre_rate = min(thre_num * change_rate, 1.0)
                    if thre_num * change_iter == self.current_iteration:
                        str = "\n"
                        str += "---" * 20
                        str += "\ncur iter: %d (%d), cur_rate: %.2f\n" % (
                            self.current_iteration,
                            change_iter,
                            thre_rate,
                        )
                        str += "---" * 20
                        print(str)
                    batch.add_field("incre_num", thre_rate)
                report, _, _ = self._forward_pass(
                    batch, enable_amp=self.enable_torch_amp
                )
                if report is None:
                    continue

                self._update_meter(report, self.meter)

                loss = self._extract_loss(report)
                self._backward(loss)
                should_break = self._logistics()

                self._run_scheduler()

                flops = metrics_reporter.get_train_flops()
                metrics_reporter.report(flops, global_step=self.current_iteration)
                metrics_reporter.reset_profile()
                self.current_iteration += 1
                self.writer.write(self.current_iteration, "debug")
                registry.register("current_iteration", self.current_iteration)
                # make sure every rank run the same iterations, so MOE module work fine;
                if self.current_iteration >= self.max_iterations:
                    break
                if should_break:
                    break

            self.current_epoch += 1

        metrics_reporter.end_profile()
        del metrics_reporter
        self.finalize()

    def _run_scheduler(self):
        update_weights = self.current_iteration % self.gradient_accumulation_steps == 0
        if self.lr_scheduler is not None and update_weights:
            self.lr_scheduler.step()

    def _forward_pass(self, batch, enable_amp=False):
        if not batch:  # Samplelist might be empty dict
            return None, None, None
        prepared_batch = self.task_loader.prepare_batch(batch)

        self.profile("Batch prepare time")
        forward_context = (
            torch.cuda.amp.autocast(enabled=True) if enable_amp else nullcontext()
        )

        with forward_context:
            # Arguments should be a dict at this point
            model_output = self.model(prepared_batch)

            if self.synchronized_loss:
                is_parallel = isinstance(self.model, nn.DataParallel) or isinstance(
                    self.model, nn.parallel.DistributedDataParallel
                )
                if "losses" not in model_output:
                    loss_func = getattr(
                        self.model.module if is_parallel else self.model, "losses"
                    )
                    model_output["losses"] = loss_func(prepared_batch, model_output)
                if "metrics" not in model_output:
                    metric_func = getattr(
                        self.model.module if is_parallel else self.model, "metrics"
                    )
                    model_output["metrics"] = metric_func(prepared_batch, model_output)

        report = Report(prepared_batch, model_output)
        self.profile("Forward time")

        return report, model_output, prepared_batch

    def _backward(self, loss):
        loss = loss / self.gradient_accumulation_steps

        if self.enable_torch_amp:
            self.scaler.scale(loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place, this should
            # be called first so that clip_gradients can take effect as usual.
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        self.profile("Backward time")

        if self.current_iteration % self.gradient_accumulation_steps != 0:
            return

        if self.should_clip_gradients:
            clip_gradients(self.model, self.current_iteration, self.writer, self.config)

        if self.enable_torch_amp:
            # If the first iteration creates NaN gradients (e.g. due to a high scaling factor and thus gradient
            # overflow), the optimizer.step() will be skipped and you might get this warning using GradScaler:
            # `optimizer.step()` before `lr_scheduler.step()` error. This is known issue, just ignore.
            # see detail:
            # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930
            self.scaler.step(self.optimizer)
            # Updates the scale for next iteration
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.profile("Optimizer time")

    def _extract_loss(self, report):
        loss_dict = report.losses
        loss = sum([loss.mean() for loss in loss_dict.values()])
        self.profile("ExtractLoss time")
        return loss

    def finalize(self):
        self.writer.write("Stepping into final validation check")
        self._try_full_validation(force=True)
        if self.checkpoint is not None:
            self.checkpoint.restore()
            self.checkpoint.finalize()
        self.inference()
        self.writer.close()

    def _update_meter(self, report, meter, sync=True):

        losses_dict = report.losses
        metrics_dict = report.metrics
        # reduce_dict -> mean == mean-reduce_dict, but we can reduce communicate traffic
        losses_dict = {key: value.mean() for key, value in losses_dict.items()}
        metrics_dict = {
            key: value.to(torch.float32).mean() for key, value in metrics_dict.items()
        }
        reduced_loss_dict = reduce_dict(losses_dict) if sync else losses_dict
        reduced_metrics_dict = reduce_dict(metrics_dict) if sync else metrics_dict

        loss_key = report.dataset_type + "/total_loss"

        with torch.no_grad():
            reduced_loss = sum([loss for loss in reduced_loss_dict.values()])
            if hasattr(reduced_loss, "item"):
                reduced_loss = reduced_loss.item()

            registry.register(loss_key, reduced_loss)

            meter_update_dict = {loss_key: reduced_loss}
            meter_update_dict.update(reduced_loss_dict)
            meter_update_dict.update(reduced_metrics_dict)
            meter.update(meter_update_dict)
        self.profile("Update Meter time")

    def _logistics(self):
        should_print = (
            self.current_iteration and self.current_iteration % self.log_interval == 0
        )
        extra = {}
        prefix = ""

        if should_print is True:
            if "cuda" in str(self.device):
                extra["max mem"] = torch.cuda.max_memory_allocated() / 1024
                extra["max mem"] //= 1024

            # display lr
            if isinstance(self.optimizer, CombinedOptimizer):
                extra["lr"] = self.optimizer.get_optimizers_lr_str()
            else:
                extra["lr"] = "|".join(
                    [
                        "{:.8f}".format(x["lr"]).rstrip("0")
                        for x in self.optimizer.param_groups
                    ]
                )

            extra.update(
                {
                    "time": self.train_timer.get_time_since_start(),
                    "eta": self._calculate_time_left(),
                }
            )

            self.train_timer.reset()

            dataset_name, single_batch_meter = self.evaluate_single_batch()
            self.meter.update_from_meter(single_batch_meter)
            prefix += dataset_name

        # Don't print train metrics if it is not log interval
        # so as to escape clutter
        self._summarize_meter(
            self.meter,
            prefix=prefix,
            extra=extra,
            should_print=should_print,
        )

        should_break = self._try_full_validation()

        return should_break

    def _try_full_validation(self, force=False):
        should_break = False

        if (
            self.current_iteration
            and self.current_iteration % self.snapshot_interval == 0
            or force
        ):
            self.writer.write("Evaluation time. Running on full validation set...")
            # Validation and Early stopping
            # Create a new meter for this case
            validation_timer = Timer()
            dataset_name, meter = self.evaluate_set(self.val_loader_list)
            extra = {"validation time": validation_timer.get_time_since_start()}

            overall_metric = self.overall_metric_evaluator.summarize()
            stop = self.early_stopping(self.current_iteration, overall_metric, meter)
            stop = bool(broadcast_scalar(stop, src=0, device=self.device))

            extra.update(self.early_stopping.get_info())

            prefix = "{}: full val".format(dataset_name)
            self._summarize_overall(overall_metric, meter, prefix=prefix, extra=extra)
            gc.collect()

            if "cuda" in str(self.device):
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

            if (
                stop > 0
            ):  # `stop` is now `int`, NCCL does not support `boolean` type's broadcasting
                self.writer.write("Early stopping activated")
                should_break = True

        return should_break

    def evaluate_single_batch(self):
        meter = Meter()

        with torch.no_grad():
            self.model.eval()
            report, _, _ = self._forward_pass(copy.deepcopy(self.first_val_batch))
            self._update_meter(report, meter)
            self.model.train()

        return report.dataset_name, meter

    def evaluate_set(self, loader_list):
        meter = Meter()

        with torch.no_grad():
            self.model.eval()
            self.overall_metric_evaluator.reset()
            for batch in tqdm(
                chain(*loader_list),
                total=self._len_of_loader_list(loader_list),
                disable=not is_main_process() or self.disable_tqdm,
            ):
                report, model_output, prepared_batch = self._forward_pass(batch)
                self._update_meter(report, meter)
                self.overall_metric_evaluator.collect(prepared_batch, model_output)
            self.model.train()

        return report.dataset_name, meter

    def interpret(self, dataset_type, use_tqdm=True):
        loader_list = getattr(self, "{}_loader_list".format(dataset_type))
        interpreter = build_interpreter(self.config, self.model)

        disable = not use_tqdm or not is_main_process()
        for batch in tqdm(
            chain(*loader_list),
            total=self._len_of_loader_list(loader_list),
            disable=disable or self.disable_tqdm,
        ):
            prepared_batch = self.task_loader.prepare_batch(batch)
            if prepared_batch is None:
                continue

            attr_info = interpreter.interpret(prepared_batch)

            interpreter.export(attr_info, prepared_batch)

    def _summarize_meter(self, meter, prefix="", extra={}, should_print=True):
        """Print batch-wise losses/metrics, used in validating a single batch in
        the training process.
        """
        if not is_main_process():
            return

        scalar_dict = meter.get_scalar_dict()
        self.writer.add_scalars(scalar_dict, registry.get("current_iteration"))

        if not should_print:
            return

        print_str = []
        if len(prefix):
            print_str += [prefix]
        print_str += ["{}/{}".format(self.current_iteration, self.max_iterations)]
        print_str += [str(meter)]
        print_str += ["{}: {}".format(key, value) for key, value in extra.items()]

        self.writer.write(meter.delimiter.join(print_str))

    def _summarize_overall(self, overall_metric, meter, prefix="", extra={}):
        """Print sample-wise metrics and batch-wise losses/metrics, used in validating
        a whole val/test set.
        """
        print_str = []
        if len(prefix):
            print_str += [prefix]
        print_str += ["{}/{}".format(self.current_iteration, self.max_iterations)]
        print_str += ["[sample-wise metrics]"]
        print_str += [
            "{}: {:.4f}".format(key, value) for key, value in overall_metric.items()
        ]
        print_str += ["[batch-wise losses/metrics]"]
        print_str += [str(meter)]
        print_str += ["{}: {}".format(key, value) for key, value in extra.items()]
        if is_main_process():
            self.writer.write(meter.delimiter.join(print_str))

    def inference(self):
        if "val" in self.run_type:
            self._inference_run("val")

        if "inference" in self.run_type or "predict" in self.run_type:
            self._inference_run("test")

        if "interpret" in self.run_type:
            self.interpret(self.run_type, True)

    def _inference_run(self, dataset_type):
        if self.config.training_parameters.evalai_inference is True:
            registry.register(constants.EVALAI_INFERENCE, True)
            self.predict_for_evalai(dataset_type)
            return

        self.writer.write("Starting inference on {} set".format(dataset_type))

        dataset_name, meter = self.evaluate_set(
            getattr(self, "{}_loader_list".format(dataset_type))
        )
        overall_metric = self.overall_metric_evaluator.summarize()
        prefix = "{}: full {}".format(dataset_name, dataset_type)
        self._summarize_overall(overall_metric, meter, prefix)

    def _calculate_time_left(self):
        time_taken_for_log = time.time() * 1000 - self.train_timer.start
        iterations_left = self.max_iterations - self.current_iteration
        num_logs_left = iterations_left / self.log_interval
        time_left = num_logs_left * time_taken_for_log

        snapshot_iteration = self.snapshot_iterations / self.log_interval
        snapshot_iteration *= iterations_left / self.snapshot_interval
        time_left += snapshot_iteration * time_taken_for_log

        return self.train_timer.get_time_hhmmss(gap=time_left)

    def profile(self, text):
        if self.not_debug:
            return
        # wait cuda kernel finish; cuda time record by current `profile`
        torch.cuda.synchronize()
        self.writer.write(text + ": " + self.profiler.get_time_since_start(), "debug")
        self.profiler.reset()

    def predict_for_evalai(self, dataset_type):
        reporter = self.task_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            message = "Starting {} inference for evalai".format(dataset_type)
            self.writer.write(message)

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm(dataloader, disable=self.disable_tqdm):
                    prepared_batch = reporter.prepare_batch(batch)
                    model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report)

            self.writer.write("Finished predicting")
            self.model.train()
