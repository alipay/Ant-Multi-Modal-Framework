# Copyright (c) 2023 Ant Group and its affiliates.
import os

import yaml
from torch.utils.data import DataLoader

from antmmf.common.batch_collator import BatchCollator
from antmmf.common.data_updater import DataUpdater
from antmmf.common.registry import registry
from antmmf.common.test_reporter import TestReporter
from antmmf.datasets.base_dataset import BaseIterableDataset
from antmmf.datasets.build import build_dataset_sampler
from antmmf.datasets.concat_dataset import AntMMFConcatDataset
from antmmf.datasets.samplers import AntmmfSampler
from antmmf.utils.distributed_utils import get_world_size


def build_collate_fn(_dataset):
    custom_collate_fn = getattr(_dataset, "collate_fn", None)
    loader_collate_fn = BatchCollator(custom_collate_fn)
    return loader_collate_fn


class TaskLoader:
    def __init__(self, config):
        self.config = config
        self.run_type = self.config.training_parameters.get("run_type", "train")
        self.task_mapping = {}
        self.loader_mapping = {}
        self.task_type = []
        self.writer = registry.get("writer")
        self._task_config = {}
        self.test_reporter = None
        self.should_not_log = self.config.training_parameters.should_not_log

    def load_task(self):

        if "train" in self.run_type:
            self.task_type += ["train", "val"]
        if "val" in self.run_type and "val" not in self.task_type:
            self.task_type += ["val"]
        if "inference" in self.run_type or "predict" in self.run_type:
            self.task_type += ["test"]
        if "interpret" in self.run_type:
            self.task_type += ["interpret"]

        assert (
            len(self.config.task_attributes.keys()) == 1
        ), "currently, only support one task."
        task_name = list(self.config.task_attributes.keys())[0]
        task_class = registry.get_task_class(task_name)
        if task_class is None:
            raise Exception(
                "[Error] %s not present in our mapping of task names" % task_name
            )

        for t_type in self.task_type:
            task_attributes = self.config["task_attributes"][task_name]
            task_attributes.defrost()
            task_attributes["dataset_type"] = t_type
            task_attributes.freeze()

            task = task_class()
            task.load(**task_attributes)
            self.task_mapping[t_type] = task

    @property
    def task_config(self):
        return self._task_config

    @task_config.setter
    def task_config(self, config):
        self._task_config = config

    def get_config(self):
        return self._task_config

    def get_test_reporter(self, dataset_type):
        task = self.task_mapping.get(dataset_type)
        assert task is not None, f"{dataset_type} task is not set"
        return TestReporter(task)

    def get_data_updater(self, dataset_type):
        task = self.task_mapping.get(dataset_type)
        assert task is not None, f"{dataset_type} task is not set"
        return DataUpdater(task)

    def _load_task_config(self, task_name):
        directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(directory, "..", "tasks", task_name, "config.yml")

        if not os.path.exists(config_path):
            print("[Warning] No config present for task %s" % task_name)
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            try:
                self._task_config = yaml.load(f)
            except yaml.YAMLError as err:
                print("[Error] Task %s's config yaml error" % task_name, err)

        return self._task_config

    def make_dataloaders(self):
        training_parameters = self.config.training_parameters
        num_workers = training_parameters.num_workers
        pin_memory = training_parameters.pin_memory

        # only make dataloaders for needed task
        for task_type, task in self.task_mapping.items():

            # One task contains several datasets, each dataset has corresponding dataloader
            dataset_name_loader_map = {}
            for name, dataset in task.datasets.items():
                extra_args = self._get_extra_args_for_dataloader(dataset)
                # chosen_dataset may be AntMMFConcatDataset or BaseDataset
                if isinstance(dataset, AntMMFConcatDataset):
                    dataset = dataset.datasets[0]

                loader_instance = DataLoader(
                    dataset=dataset,
                    pin_memory=pin_memory,
                    collate_fn=build_collate_fn(dataset),
                    num_workers=num_workers,
                    **extra_args,
                )
                loader_instance.dataset_type = task_type
                dataset_name_loader_map[name] = loader_instance

            self.loader_mapping[task_type] = dataset_name_loader_map

    def _get_extra_args_for_dataloader(self, dataset):
        training_parameters = self.config.training_parameters
        extra_args = {"shuffle": False}

        if dataset.dataset_type != "test" and not isinstance(
            dataset, BaseIterableDataset
        ):
            extra_args["shuffle"] = True

        if (
            training_parameters.local_rank is not None
            and training_parameters.distributed
        ):
            # for distributed env, using antmmf.datasets.samplers.DistributedSampler as default sampler
            sampler_config = training_parameters.distributed_sampler

        else:  # for local or data-parallel
            sampler_config = training_parameters.sampler

        sampler = build_dataset_sampler(
            dataset, sampler_config, default_config={"shuffle": extra_args["shuffle"]}
        )

        if sampler is not None:  # indicate sampler
            assert isinstance(sampler, AntmmfSampler)
            if sampler.is_batch_sampler():
                # batch_sampler mutually exclusive with :attr:`batch_size`,
                # :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
                extra_args["batch_sampler"] = sampler
            else:
                # :attr:`shuffle` must be ``False``.
                extra_args["sampler"] = sampler

            # For class_balance_sampler, set drop_last flag during training
            if (
                sampler_config.type == "class_balance_sampler"
                and dataset.dataset_type != "test"
            ):
                extra_args["drop_last"] = True

            # Shuffle is mutually exclusive with sampler, let sampler take care of shuffle and pop from main args
            extra_args.pop("shuffle")
            setattr(self, f"{dataset.dataset_type}_{dataset.name}_sampler", sampler)

        batch_size = training_parameters.batch_size

        if dataset.dataset_type != "train":
            # handle cases when training batch_size does not match test batch_size
            batch_size = training_parameters.get("test_batch_size", batch_size)

        world_size = get_world_size()

        if batch_size % world_size != 0:
            raise RuntimeError(
                f"Batch size {batch_size} must be divisible by number of GPUs {world_size} used."
            )

        if "batch_sampler" not in extra_args:
            # batch_sampler mutually exclusive with :attr:`batch_size`
            extra_args["batch_size"] = batch_size // world_size

        return extra_args

    def update_registry_for_model(self, config):
        for task_type, task in self.task_mapping.items():
            task.update_registry_for_model(config)

    def clean_config(self, config):
        for task_type, task in self.task_mapping.items():
            task.clean_config(config)

    def prepare_batch(self, batch, *args, **kwargs):
        if batch is None:
            return None
        return self.task_mapping[batch["dataset_type"]].prepare_batch(batch)

    def verbose_dump(self, report, *args, **kwargs):
        if self.config.training_parameters.verbose_dump:
            dataset_type = report.dataset_type
            self.task_mapping[dataset_type].verbose_dump(report, *args, **kwargs)

    def seed_sampler(self, task_type, seed):
        training_parameters = self.config.training_parameters
        if (
            training_parameters.local_rank is not None
            and training_parameters.distributed
        ):
            task = self.task_mapping[task_type]
            for dataset in task.datasets.values():
                sampler = getattr(self, f"{task_type}_{dataset.name}_sampler")
                assert hasattr(
                    sampler, "set_epoch"
                ), "Can't seed without `set_epoch` method"
                sampler.set_epoch(seed)
