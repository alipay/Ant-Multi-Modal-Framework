# Copyright (c) 2023 Ant Group and its affiliates.
"""
Tasks come above datasets in hierarchy level. In case you want to
implement a new task, you need to inherit ``BaseTask`` class. You need
to implement ``_get_available_datasets`` and ``_preprocess_item`` functions
to complete the implementation. You can check the source to see if you need
to override any other methods like ``prepare_batch``.

Check example of ``VQATask`` here_.

Example::

    from antmmf.tasks.base_task import BaseTask
    from antmmf.common.registry import registry


    @registry.register_task("my")
    class MyTask(BaseTask):
        def __init__(self):
            super().__init__("my")

        def _get_available_datasets(self):
            return ["my"]

        def _preprocess_item(self):
            item.text = None
            return item

.. _here: https://github.com/facebookresearch/pythia/blob/v0.3/pythia/tasks/vqa/vqa_task.py
"""

import sys

import numpy as np
from antmmf.common.registry import registry
from torch.utils.data import Dataset, IterableDataset


class BaseTask(Dataset):
    """
    BaseTask that task classes need to inherit in order to create a new task.

    Users must implement ``_get_available_datasets`` and ``_preprocess_item``
    in order to complete implementation.

    Args:
        task_name (str): Name of the task with which it will be registered
    """

    def __init__(self, task_name):
        super(BaseTask, self).__init__()
        self.task_name = task_name
        self.writer = registry.get("writer")

        self.task_annotations = {}

    def load(self, **opts):
        self.opts = opts

        self.datasets = {}
        self.datasets_name = []
        self.builders = {}
        self.dataset_type = self.opts.get("dataset_type", "train")
        available_datasets = self._get_available_datasets()

        self.total_length = 0
        self.per_dataset_lengths = []
        self.num_datasets = 0
        for dataset_key in self.opts["dataset_attributes"]:
            if dataset_key in available_datasets:
                builder_class = registry.get_builder_class(dataset_key)

                assert (
                    builder_class is not None
                ), f"No builder class found for dataset {dataset_key}."

                builder_instance = builder_class()

                if dataset_key in self.opts["dataset_attributes"]:
                    attributes = self.opts["dataset_attributes"][dataset_key]
                else:
                    self.writer.write(
                        "Dataset %s is missing from dataset_attributes in config."
                        % dataset_key,
                        "error",
                    )
                    sys.exit(1)

                builder_instance.build(self.dataset_type, attributes)
                dataset_instance = builder_instance.load(self.dataset_type, attributes)
                # TODO: add dataset_loader, supporting customizing dataset loader
                if dataset_instance is None:
                    continue

                dataset_name = dataset_instance._name
                self.datasets_name.append(dataset_name)
                self.builders[dataset_name] = builder_instance
                self.datasets[dataset_name] = dataset_instance
                assert isinstance(
                    dataset_instance, Dataset
                ), "The dataset_instance should be Dataset. If it is IterableDatset, use BaseIterableTask instead"
                self.per_dataset_lengths.append(len(dataset_instance))
                self.total_length += len(dataset_instance)
            else:
                raise Exception(
                    "Dataset %s is not a valid dataset for task %s. Skipping"
                    % (dataset_key, self.task_name)
                )

        self.num_datasets = len(self.datasets)
        self.dataset_probabilities = [
            1.0 / self.num_datasets for _ in range(self.num_datasets)
        ]
        sampling = self.opts.get("dataset_size_proportional_sampling", None)

        if sampling is True:
            self.dataset_probabilities = self.per_dataset_lengths[:]
            self.dataset_probabilities = [
                prob / self.total_length for prob in self.dataset_probabilities
            ]

        self.change_dataset()

    def _get_available_datasets(self):
        """Set available datasets for this task here.
        Override in your child task class
        Temporary solution, later we will use decorators to easily register
        datasets with a task

        Returns:
            List - List of available datasets for this particular task
        """
        return []

    def get_datasets(self):
        return self.datasets

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        idx = idx % self.per_dataset_lengths[self.dataset_choice]

        item = self.chosen_dataset[idx]

        return self._preprocess_item(item)

    def change_dataset(self):
        self.dataset_choice = np.random.choice(
            self.num_datasets, 1, p=self.dataset_probabilities
        )[0]
        chosen_dataset_name = self.datasets_name[self.dataset_choice]
        self.chosen_dataset = self.datasets[chosen_dataset_name]

    def select_dataset(self, dataset_name):
        self.chosen_dataset = self.datasets[dataset_name]

    def verbose_dump(self, *args, **kwargs):
        self.chosen_dataset.verbose_dump(*args, **kwargs)

    def prepare_batch(self, batch):
        return self.chosen_dataset.prepare_batch(batch)

    def _preprocess_item(self, item):
        """Preprocess an item to be returned from __getitem__.
        Override in your child task class, so you have control on what you are
        returning

        Args:
            item (Sample): Sample returned by a particular dataset

        Returns:
            Sample: Preprocessed item
        """
        raise NotImplementedError("This task doesn't implement preprocess_item method")

    def update_registry_for_model(self, config):
        """
        Use this if there is some specific configuration required by model
        which must be inferred at runtime.
        """
        for builder in self.builders.values():
            builder.update_registry_for_model(config)

    def init_args(self, parser):
        parser.add_argument_group("General Task Arguments")
        parser.add_argument(
            "-dsp",
            "--dataset_size_proportional_sampling",
            type=bool,
            default=0,
            help="Pass if you want to sample from"
            " dataset according to its size. Default: Equal "
            " weighted sampling",
        )

        # TODO: Figure out later if we want to init args from datasets
        # self._init_args(parser)

    def _init_args(self, parser):
        """Override this function to add extra parameters to
        parser in your child task class.

        Parameters
        ----------
        parser : ArgumentParser
            Original parser object passed from the higher level classes like
            trainer

        Returns
        -------
        type
            Description of returned object.

        """
        for builder in self.builders.values():
            builder.init_args(parser)

    def clean_config(self, config):
        """
        Override this in case you want to clean the config you updated earlier
        in update_registry_for_model
        """
        return config

    def get_annotations(self, annotation_type):
        """
        Annotations of task, consists of all datasets' annotations
        """
        if annotation_type not in self.task_annotations:
            self.task_annotations[annotation_type] = []
            for dataset in self.datasets:
                assert hasattr(dataset, "get_annotations")
                self.task_annotations[annotation_type].extend(
                    dataset.get_annotations(annotation_type)
                )
        return self.task_annotations[annotation_type]


class BaseIterableTask(IterableDataset):
    """
    BaseIterableTask that task classes need to inherit in order to create a new iterable task.

    Users must implement ``_get_available_datasets`` and ``_preprocess_item``
    in order to complete implementation.

    Args:
        task_name (str): Name of the task with which it will be registered
    """

    def __init__(self, task_name):
        super(BaseIterableTask, self).__init__()
        self.task_name = task_name
        self.writer = registry.get("writer")

        self.task_annotations = {}

    def load(self, **opts):
        self.opts = opts

        self.datasets = {}
        self.datasets_name = []
        self.builders = {}
        self.dataset_type = self.opts.get("dataset_type", "train")
        available_datasets = self._get_available_datasets()

        self.total_length = 0
        self.num_datasets = 0
        for dataset_key in self.opts["dataset_attributes"]:
            if dataset_key in available_datasets:
                builder_class = registry.get_builder_class(dataset_key)

                assert (
                    builder_class is not None
                ), f"No builder class found for dataset {dataset_key}."

                builder_instance = builder_class()

                if dataset_key in self.opts["dataset_attributes"]:
                    attributes = self.opts["dataset_attributes"][dataset_key]
                else:
                    self.writer.write(
                        "Dataset %s is missing from dataset_attributes in config."
                        % dataset_key,
                        "error",
                    )
                    sys.exit(1)

                builder_instance.build(self.dataset_type, attributes)
                dataset_instance = builder_instance.load(self.dataset_type, attributes)
                # TODO: add dataset_loader, supporting customizing dataset loader
                if dataset_instance is None:
                    continue
                assert isinstance(
                    dataset_instance, IterableDataset
                ), "dataset_instance should be IterableDataset"
                dataset_name = dataset_instance._name
                self.datasets_name.append(dataset_name)
                self.builders[dataset_name] = builder_instance
                self.datasets[dataset_name] = dataset_instance
                self.total_length += len(dataset_instance)
            else:
                raise Exception(
                    "Dataset %s is not a valid dataset for task %s. Skipping"
                    % (dataset_key, self.task_name)
                )

        self.num_datasets = len(self.datasets)
        self.dataset_probabilities = [
            1.0 / self.num_datasets for _ in range(self.num_datasets)
        ]
        sampling = self.opts.get("dataset_size_proportional_sampling", None)

        if sampling is True:
            self.dataset_probabilities = self.per_dataset_lengths[:]
            self.dataset_probabilities = [
                prob / self.total_length for prob in self.dataset_probabilities
            ]

        self.change_dataset()

    def _get_available_datasets(self):
        """Set available datasets for this task here.
        Override in your child task class
        Temporary solution, later we will use decorators to easily register
        datasets with a task

        Returns:
            List - List of available datasets for this particular task
        """
        return []

    def get_datasets(self):
        return self.datasets

    def __len__(self):
        return self.total_length

    def __iter__(self):
        for item in iter(self.chosen_dataset):
            yield self._preprocess_item(item)

    def change_dataset(self):
        self.dataset_choice = np.random.choice(
            self.num_datasets, 1, p=self.dataset_probabilities
        )[0]
        chosen_dataset_name = self.datasets_name[self.dataset_choice]
        self.chosen_dataset = self.datasets[chosen_dataset_name]

    def select_dataset(self, dataset_name):
        self.chosen_dataset = self.datasets[dataset_name]

    def verbose_dump(self, *args, **kwargs):
        self.chosen_dataset.verbose_dump(*args, **kwargs)

    def prepare_batch(self, batch):
        return self.chosen_dataset.prepare_batch(batch)

    def _preprocess_item(self, item):
        """Preprocess an item to be returned from __getitem__.
        Override in your child task class, so you have control on what you are
        returning

        Args:
            item (Sample): Sample returned by a particular dataset

        Returns:
            Sample: Preprocessed item
        """
        raise NotImplementedError("This task doesn't implement preprocess_item method")

    def update_registry_for_model(self, config):
        """
        Use this if there is some specific configuration required by model
        which must be inferred at runtime.
        """
        for builder in self.builders.values():
            builder.update_registry_for_model(config)

    def init_args(self, parser):
        parser.add_argument_group("General Task Arguments")
        parser.add_argument(
            "-dsp",
            "--dataset_size_proportional_sampling",
            type=bool,
            default=0,
            help="Pass if you want to sample from"
            " dataset according to its size. Default: Equal "
            " weighted sampling",
        )

        # TODO: Figure out later if we want to init args from datasets
        # self._init_args(parser)

    def _init_args(self, parser):
        """Override this function to add extra parameters to
        parser in your child task class.

        Parameters
        ----------
        parser : ArgumentParser
            Original parser object passed from the higher level classes like
            trainer

        Returns
        -------
        type
            Description of returned object.

        """
        for builder in self.builders.values():
            builder.init_args(parser)

    def clean_config(self, config):
        """
        Override this in case you want to clean the config you updated earlier
        in update_registry_for_model
        """
        return config

    def get_annotations(self, annotation_type):
        """
        Annotations of task, consists of all datasets' annotations
        """
        if annotation_type not in self.task_annotations:
            self.task_annotations[annotation_type] = []
            for dataset in self.datasets:
                assert hasattr(dataset, "get_annotations")
                self.task_annotations[annotation_type].extend(
                    dataset.get_annotations(annotation_type)
                )
        return self.task_annotations[annotation_type]
