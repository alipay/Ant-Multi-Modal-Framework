# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from typing import Union
from antmmf.utils.distributed_utils import is_dist_avail_and_initialized
from antmmf.datasets.utils import TextReader, ODPSReader


class KgrlDatabase(torch.utils.data.IterableDataset):
    def __init__(
        self,
        sampler_conf,
        dataset_type: str,
        seeds: Union[TextReader, ODPSReader] = None,
    ):
        assert sampler_conf is not None
        self.sampler_conf = sampler_conf.to_dict()
        self.sampler_conf["compact_log"] = True

        if not is_dist_avail_and_initialized():
            self.sampler_conf["rank"] = 0

        if dataset_type in ["valid", "test"]:
            self.sampler_conf["gen_data_conf"]["random"] = False

        self.seeds = seeds if seeds is not None else None
        self.dataset_type = dataset_type
        self.buffer_size = sampler_conf.gen_data_conf.buffer_size
        self.sampler_type = self.sampler_conf["type"]

    def __iter__(self):
        from kgrl.data.sampler import KGStateCacheBaseSampler
        from kgrl.conf import KgrlConstants
        from libkg_client import MurmurHash64

        sampler = KGStateCacheBaseSampler.from_params(self.sampler_conf)
        sampler.connect()

        if self.seeds is None:
            if self.dataset_type == "train":
                sampler.set_dataset_type(KgrlConstants.TRAIN_DATASET_TYPE)
            elif self.dataset_type == "valid":
                sampler.set_dataset_type(KgrlConstants.VALIDATION_DATASET_TYPE)
            else:
                sampler.set_dataset_type(KgrlConstants.TEST_DATASET_TYPE)
        else:
            sampler.set_dataset_type(KgrlConstants.TRAIN_DATASET_TYPE)

        if self.seeds is not None:
            seeds = iter(self.seeds)
            while True:
                # sample sub-graphs in batch-mode
                cur_seeds = []
                try:
                    for _ in range(self.buffer_size):
                        cur_seeds.append(next(seeds))
                except StopIteration:
                    pass

                if len(cur_seeds) == 0:
                    break
                for seed in cur_seeds:
                    if self.sampler_type == "edge":
                        seed["src"] = MurmurHash64.hash(
                            seed["src"] + "_" + seed["src_type"].lower()
                        )
                        seed["dst"] = MurmurHash64.hash(
                            seed["dst"] + "_" + seed["dst_type"].lower()
                        )
                    elif self.sampler_type == "node":
                        seed["seed"] = MurmurHash64.hash(
                            seed["seed"] + "_" + seed["seed_type"].lower()
                        )

                index = 0
                for sample in sampler.gen_data(cur_seeds):
                    for key, value in cur_seeds[index].items():
                        if key not in sample:
                            sample[key] = value
                    index += 1
                    yield sample
        else:
            while True:
                try:
                    yield sampler.gen_data(None)
                except StopIteration:
                    break
