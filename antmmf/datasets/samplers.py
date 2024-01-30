# Copyright (c) 2023 Ant Group and its affiliates.

import copy
import math
import random
from collections import defaultdict
from typing import Type

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset

from antmmf.common.registry import registry
from antmmf.datasets.base_dataset import BaseDataset


class AntmmfSampler(Sampler):
    r"""Base class for all antmmf samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of data_source elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: data_source in antmmf indicates an instance of antmmf.tasks.base_dataset.BaseDataset class
    """

    def __init__(self, data_source, *args, **kwargs):
        assert isinstance(data_source, BaseDataset) or isinstance(
            data_source, Dataset
        ), f"{data_source} must be instance of BaseDataset or Dataset"
        self.data_source = data_source

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def is_batch_sampler(self):
        raise NotImplementedError


@registry.register_sampler("sequential_sampler")
class SequentialSampler(AntmmfSampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, base_dataset: Type[Dataset]):
        self.data_source = base_dataset

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)

    def is_batch_sampler(self):
        return False


@registry.register_sampler("random_sampler")
class RandomSampler(AntmmfSampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (BaseTask): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, base_task, replacement=False, num_samples=None, **kwargs):
        super().__init__(base_task, **kwargs)
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

        if self._num_samples is not None and not replacement:
            raise ValueError(
                "With replacement=False, num_samples should not be specified, "
                "since a random permute will be performed."
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(
                torch.randint(
                    high=n, size=(self.num_samples,), dtype=torch.int64
                ).tolist()
            )
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples

    def is_batch_sampler(self):
        return False


@registry.register_sampler("distributed_sampler")
class DistributedSampler(AntmmfSampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        In antmmf, Sampler is assumed to sample from instance of antmmf.tasks.base_task.BaseTask
    Arguments:
        base_task: instance of antmmf.tasks.base_task.BaseTask, used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(
        self, base_dataset, num_replicas=None, rank=None, shuffle=True, **kwargs
    ):
        super().__init__(base_dataset, **kwargs)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.data_source) * 1.0 / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.data_source), generator=g).tolist()
        else:
            indices = torch.arange(len(self.data_source)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def is_batch_sampler(self):
        return False


@registry.register_sampler("class_balance_sampler")
class RandomClassSampler(AntmmfSampler):
    """
    Randomly sample N classes, then for each class,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - base_task (Dataset): dataset to sample from.
    - num_instance (int): number of instances per class.
    """

    def __init__(self, base_dataset, num_instance=4, **kwargs):
        super().__init__(base_dataset, **kwargs)
        self.num_instance = num_instance
        self.index_dic = defaultdict(list)

        self.img_labels = []
        self.img_labels += base_dataset.get_annotations("labels")

        for index, label in enumerate(self.img_labels):
            self.index_dic[label].append(index)
        self.labels = list(self.index_dic.keys())
        self.num_classes = len(self.labels)

        # compute number of examples in an epoch
        self.length = 0
        for label in self.labels:
            idxs = self.index_dic[label]
            num = len(idxs)
            if num < self.num_instance:
                num = self.num_instance
            self.length += num - num % self.num_instance

    def __iter__(self):
        list_container = []

        for label in self.labels:
            idxs = copy.deepcopy(self.index_dic[label])
            if len(idxs) < self.num_instance:
                idxs = np.random.choice(idxs, size=self.num_instance, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instance:
                    list_container.append(batch_idxs)
                    batch_idxs = []

        random.shuffle(list_container)

        ret = []
        for batch_idxs in list_container:
            ret.extend(batch_idxs)

        return iter(ret)

    def __len__(self):
        return self.length

    def is_batch_sampler(self):
        return False
