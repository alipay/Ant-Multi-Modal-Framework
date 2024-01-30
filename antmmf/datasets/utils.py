# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import math
from typing import List, Dict
from itertools import islice
from torch.utils.data import get_worker_info
from antmmf.common.configuration import Configuration
from antmmf.utils.distributed_utils import get_world_size, get_rank


def text_classification_online_process(
    text: List[str], max_len: int, tokenizer
) -> Dict:
    """
    Used for the task of text classification, and it also can be used as a tokenize function.

    Args:
          text (list[str]): for example: ["蚂蚁金服内容理解"]
          max_len (int): maximum length of tokens
          tokenizer

    Returns:
        a dict
    """
    input_ids = (
        [tokenizer.cls_token_id]
        + tokenizer.encode(text[0], add_special_tokens=False)
        + [tokenizer.sep_token_id]
    )
    if len(text) == 2:
        input_ids += tokenizer.encode(text[1], add_special_tokens=False) + [
            tokenizer.sep_token_id
        ]
    token_type_ids = [0] * len(input_ids)
    assert len(input_ids) == len(token_type_ids)
    mask = [1] * len(input_ids)
    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + (
            [tokenizer.pad_token_type_id] * padding_length
        )
    else:
        input_ids = input_ids[:max_len]
        mask = mask[:max_len]
        token_type_ids = token_type_ids[:max_len]
    return {
        "ids": input_ids,  # for backward compatibility, keep using ids as keys
        "text": input_ids,
        "mask": mask,
        "token_type_ids": token_type_ids,
        "orig_text": text,
    }


def block_read(files, size=65536):
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


class TextReader:
    """
    Read seeds of knowledge graph from text file, and the text file is like a table,
    for each line, it is a record of the corresponding odps table, and the table is
    downloaded by odpscmd.

    This class will help us to parse the downloaded text file to records, and return
    an iterator to us.

    The sharding strategy is applied according to the num_worker of dataloader and
    the number of GPUs. For example, if there are 12800 training or validation samples, 4 worker for a
    dataloader, and 2 gpus, then the table will be seperated to 8 partitions, each partition have 1600 samples.

    Args:
        file_path (str): path of the seeds file.
        field_names (list): schema of the seeds table.
        field_delimiter (str): the values in each record will be converted to strings
          and joined by a delimiter, therefore, we need the delimiter to parse each
          line in the downloaded text file.

    Returns:
        An iterator

    """

    def __init__(
        self,
        file_path: str,
        field_names: List[str],
        field_delimiter: str = None,
    ):
        self.file_path = file_path
        self.field_names = field_names
        self.field_delimiter = field_delimiter
        self._size = None
        self.rank = get_rank()
        self.world_size = get_world_size()

    def __len__(self) -> int:
        if self._size is None:
            with open(self.file_path, "r", encoding="utf-8") as fd:
                self._size = sum(bl.count("\n") for bl in block_read(fd))

        return self._size

    def __iter__(self):
        # if not use multiprocessing, then get_worker_info() will return `None`
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        with open(self.file_path, "r", encoding="utf-8") as fd:
            for line in islice(
                fd,
                self.rank * num_workers + worker_id,
                None,
                self.world_size * num_workers,
            ):
                fields = line.strip().split(self.field_delimiter)
                if len(fields) != len(self.field_names):
                    raise RuntimeError(
                        f"parse `{line}` failed, the length fields it parsed is not equal to the "
                        f"length of `{self.field_names}`, got `{len(fields)}`, expect `{len(self.field_names)}`."
                    )
                yield {key: value for key, value in zip(self.field_names, fields)}


class ODPSReader:
    def __init__(
        self,
        table_name: str,
        partition: str = None,
        mapping: Configuration = None,
        max_retry_times: int = 10,
        retry_interval: int = 5000,
        cache_size: int = 1000,
    ):
        """
        Read data from ODPS, and it will help us to get each record in the ODPS table stably. The sharding strategy
        is applied according to the num_worker of dataloader and the number of GPUs. For example, if there are 12800
        training or validation samples, 4 worker for a dataloader, and 2 gpus, then the table will be seperated to 8
        partitions, each partition have 1600 samples.

        Args:
            table_name: the name of an ODPS table, such as ant_p13n_dev.ranghou_mmf_train
            partition: partition of the table, the default value is `None`
            mapping: transform the schema of the table
            max_retry_times: the max retry times for some ODPS operation to solve the problem of network dithering
            retry_interval: the interval time for retry (ms)
            cache_size: the cache size for ODPS, default value is 1000, larger value can enhance the reliability of
            the reader, but it will increase the memory usage and startup time.

        Returns:
            An iterator

        """
        self.table_name = table_name
        self.partition = partition
        self.mapping = mapping
        self.max_retry_times = max_retry_times
        self.retry_interval = retry_interval
        self.cache_size = cache_size
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.cache = []

    def __len__(self) -> int:
        reader = self.construct_reader()
        return int(reader.count)

    def construct_reader(self):
        from retrying import retry
        from pypai.commons.utils import env_utils

        @retry(
            stop_max_attempt_number=self.max_retry_times, wait_fixed=self.retry_interval
        )
        def _construct_read():
            odps = env_utils.get_odps_instance()
            return odps.get_table(name=self.table_name).open_reader(
                partition=self.partition
            )

        return _construct_read()

    def __iter__(self):
        from retrying import retry

        count = len(self)
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        per_worker = int(math.ceil(count / num_workers / self.world_size))
        start = (self.rank * num_workers + worker_id) * per_worker
        read_count = min(count - start, per_worker)
        status = {"already_read": 0}

        @retry(
            stop_max_attempt_number=self.max_retry_times, wait_fixed=self.retry_interval
        )
        def fill_cache():
            already_read = status["already_read"]
            cur_count = min(
                self.cache_size - len(self.cache), read_count - already_read
            )
            for _record in self.construct_reader().read(
                start=start + already_read, count=cur_count
            ):
                status["already_read"] += 1
                self.cache.append(_record)

        while status["already_read"] < read_count:
            self.cache = []
            fill_cache()

            for record in self.cache:
                record = dict(record)
                if self.mapping is not None:
                    for src_key, dst_key in self.mapping.items():
                        record[dst_key] = record.pop(src_key)
                yield record
