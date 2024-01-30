# Copyright (c) 2023 Ant Group and its affiliates.
import csv
import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from antmmf.common import constants
from antmmf.common.batch_collator import BatchCollator
from antmmf.common.constants import (
    BATCH_SIZE_STR,
    CONFIG_STR,
    EXPERIMENT_NAME_STR,
    NUM_WORKERS_STR,
    REPORT_FOLDER_STR,
    REPORT_FORMAT_STR,
    SAMPLER_STR,
    SHUFFLE_STR,
    TRAINING_PARAMETERS_STR,
    WRITER_STR,
)
from antmmf.common.registry import registry
from antmmf.utils.distributed_utils import (
    all_gather,
    gather_tensor,
    get_world_size,
    is_main_process,
)
from antmmf.utils.general import (
    ckpt_name_from_core_args,
    foldername_from_config_override,
    jsonl_dump,
)
from antmmf.utils.timer import Timer


class TestReporter:
    def __init__(self, task_instance):
        self.test_task = task_instance
        self.task_type = task_instance.dataset_type
        self.config = registry.get(CONFIG_STR)
        self.writer = registry.get(WRITER_STR)
        self.report = []
        self.timer = Timer()
        self.training_parameters = self.config[TRAINING_PARAMETERS_STR]
        self.num_workers = self.training_parameters[NUM_WORKERS_STR]
        self.batch_size = self.training_parameters[BATCH_SIZE_STR]
        self.report_folder_arg = self.config.get(REPORT_FOLDER_STR, None)
        self.experiment_name = self.training_parameters.get(EXPERIMENT_NAME_STR, "")
        self.report_format = self.training_parameters.get(REPORT_FORMAT_STR, "json")

        self.datasets = [ds for ds in self.test_task.get_datasets().values()]
        self.total_len = sum([len(ds) for ds in self.datasets])
        # whether to write evalai inference predictions into multiple sub files.
        self.split_evalai = self.total_len > float(
            self.training_parameters.evalai_max_predictions_per_file
        )

        self.current_dataset_idx = -1
        self.current_dataset = self.datasets[self.current_dataset_idx]

        self.save_dir = self.training_parameters.get("save_dir", "./save")
        self.report_folder = ckpt_name_from_core_args(self.config)
        self.report_folder += foldername_from_config_override(self.config)

        self.report_folder = os.path.join(self.save_dir, self.report_folder)
        self.report_folder = os.path.join(self.report_folder, "reports")

        if self.report_folder_arg is not None:
            self.report_folder = self.report_folder_arg

        if not os.path.exists(self.report_folder):
            os.makedirs(self.report_folder)

    def next_dataset(self):
        if self.current_dataset_idx >= 0:
            self.flush_report()

        self.current_dataset_idx += 1

        if self.current_dataset_idx == len(self.datasets):
            return False
        else:
            self.current_dataset = self.datasets[self.current_dataset_idx]
            self.writer.write("Predicting for " + self.current_dataset.name)
            return True

    def csv_dump(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            title = self.report[0].keys()
            cw = csv.DictWriter(f, title, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            cw.writeheader()
            cw.writerows(self.report)

    def json_dump(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

    def get_output_path(self):
        name = self.current_dataset.name
        time_format = "%Y-%m-%dT%H:%M:%S"
        time = self.timer.get_time_hhmmss(None, format=time_format)

        filename = name + "_"

        if len(self.experiment_name) > 0:
            filename += self.experiment_name + "_"

        filename += self.task_type + "_"

        filename += time

        # decide suffix
        if self.report_format == "csv":
            suffix = ".csv"
        elif self.report_format == "jsonl":
            suffix = ".jsonl"
        else:
            suffix = ".json"

        filepath = os.path.join(self.report_folder, filename + suffix)

        return filepath

    def get_split_file_path(self, filepath):
        if not self.split_evalai:
            return filepath
        evalai_index = 0
        output_file = filepath + "_%s" % evalai_index
        while osp.exists(output_file):
            evalai_index += 1
            output_file = filepath + "_%s" % evalai_index
        return output_file

    def flush_report(self):
        if not is_main_process():
            return

        filepath = (
            self.training_parameters.get("evalai_inference_file")
            or self.get_output_path()
        )
        flush_file = self.get_split_file_path(filepath)

        if len(self.report) == 0:
            return

        self.report = self.current_dataset.align_evalai_report_order(self.report)

        if self.report_format == "csv":
            self.csv_dump(flush_file)
        elif self.report_format == "jsonl":
            jsonl_dump(self.report, flush_file)
        else:
            self.json_dump(flush_file)

        self.writer.write(
            "Wrote evalai predictions for %s to %s"
            % (self.current_dataset.name, os.path.abspath(flush_file))
        )
        self.report = []

    def get_dataloader(self):
        other_args = self._add_extra_args_for_dataloader()

        dataset = self.current_dataset
        from antmmf.datasets.concat_dataset import AntMMFConcatDataset

        # chosen_dataset may be AntMMFConcatDataset or BaseDataset
        if isinstance(self.current_dataset, AntMMFConcatDataset):
            dataset = self.current_dataset.datasets[0]

        def build_collate_fn(_dataset):
            if hasattr(_dataset, "collate_fn"):
                custom_collate_fn = getattr(_dataset, "collate_fn", None)

                def loader_collate_fn(batch):
                    return BatchCollator()(custom_collate_fn(batch))

            else:
                loader_collate_fn = BatchCollator()
            return loader_collate_fn

        return DataLoader(
            dataset=self.current_dataset,
            collate_fn=build_collate_fn(dataset),
            num_workers=self.num_workers,
            pin_memory=self.config.training_parameters.pin_memory,
            **other_args
        )

    def _add_extra_args_for_dataloader(self, other_args={}):
        training_parameters = self.config.training_parameters

        if (
            training_parameters.local_rank is not None
            and training_parameters.distributed
        ):
            other_args[SAMPLER_STR] = DistributedSampler(self.current_dataset)
        else:
            other_args[
                SHUFFLE_STR
            ] = False  # update to false, keep same order by default

        # use test_batch_size prior to batch_size for evalai_inference
        batch_size = training_parameters.get(
            "test_batch_size", training_parameters.batch_size
        )

        world_size = get_world_size()

        if batch_size % world_size != 0:
            raise RuntimeError(
                "Batch size {} must be divisible by number "
                "of GPUs {} used.".format(batch_size, world_size)
            )

        other_args[BATCH_SIZE_STR] = batch_size // world_size

        return other_args

    def prepare_batch(self, batch):
        return self.current_dataset.prepare_batch(batch)

    def __len__(self):
        return len(self.current_dataset)

    def __getitem__(self, idx):
        return self.current_dataset[idx]

    def add_to_report(self, report):
        report_keys = report.keys()
        for key in report_keys:
            if isinstance(report[key], torch.Tensor):
                tdim = report[key].ndim
                # do distributed reduce
                report[key] = gather_tensor(report[key])

                if tdim >= 2:
                    # detection results: pred_boxes / pred_logits: batch_size, num_boxes, 4 / num_classes
                    if key in constants.DETECTION_RESULTS_KEYS:
                        continue
                    # classification results: reduce to 2-dim, [batch_size, logits/probs]
                    report[key] = report[key].view(-1, report[key].size(-1))
                else:
                    report[key] = report[key].view(-1)

            elif isinstance(report[key], list):
                try:
                    report[key] = torch.Tensor(report[key])
                # report[key] is a list of strings
                except (ValueError, TypeError):
                    if key in constants.POSSIBLE_IMAGE_NAME_STRS:
                        report[key] = [
                            v
                            for reduce_val in all_gather(report[key])
                            for v in reduce_val
                        ]
            else:  # ignore
                continue

        if not is_main_process():
            return
        results = self.current_dataset.format_for_evalai(report)
        self.report = self.report + results
        if len(self.report) >= float(
            self.training_parameters.evalai_max_predictions_per_file
        ):
            self.flush_report()
