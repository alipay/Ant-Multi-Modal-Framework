# Copyright (c) 2023 Ant Group and its affiliates.
import os
import torchvision
from antmmf.common.constants import (
    ID_STR,
    QUESTION_ID_STR,
)
from antmmf.utils.distributed_utils import gather_tensor, is_main_process
from antmmf.common.report import Report
from .test_reporter import TestReporter
from antmmf.utils.general import (
    ckpt_name_from_core_args,
    foldername_from_config_override,
    jsonl_dump,
)


class DataUpdater(TestReporter):
    def __init__(self, task_instance):
        super().__init__(task_instance)
        self.acc_report = []

        self.report_folder = ckpt_name_from_core_args(self.config)
        self.report_folder += foldername_from_config_override(self.config)

        self.report_folder = os.path.join(self.save_dir, self.report_folder)
        self.report_folder = os.path.join(self.report_folder, "img")

        if not os.path.exists(self.report_folder):
            os.makedirs(self.report_folder)

        self.writer.write("Save updated data in " + self.report_folder)
        self.report_format = ".jsonl"

    def add_to_report(self, report: Report):
        # update data
        report.logits = gather_tensor(report.logits).view(-1, report.logits.size(-1))
        if ID_STR in report:
            report.id = gather_tensor(report.id).view(-1)
        if QUESTION_ID_STR in report:
            report.question_id = gather_tensor(report.question_id).view(-1)

        if not is_main_process():
            return

        results = self.current_dataset.format_for_overwrite_obs_labels(report)
        self.report = self.report + results

    def flush_report(self):
        if not is_main_process():
            return

        self.report = self.acc_report
        if len(self.report) > 0:
            filepath = self.get_output_path()
            jsonl_dump(self.report, filepath)

            self.writer.write(
                f"Wrote updated annotation for {self.current_dataset.name} to {os.path.abspath(filepath)}"
            )

        self.report = []
        self.acc_report = []

    def flush_intermediate_report(self):
        if not is_main_process():
            return

        for report in self.report:
            image_id, image, image_delta = (
                report["id"],
                report["image"],
                report["image_delta"],
            )
            for img, suffix in zip([image, image_delta], [".png", "_delta.png"]):
                image_obj = torchvision.transforms.ToPILImage()(img.cpu().detach())
                image_obj.save(f"{self.report_folder}/{image_id}" + suffix)

        # remove those have been saved to images
        for idx, rep in enumerate(self.report):
            for k in ["image", "image_delta", "org_image"]:
                del rep[k]
            self.acc_report.append(rep)

        self.report = []
