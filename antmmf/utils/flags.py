# Copyright (c) 2023 Ant Group and its affiliates.
import argparse


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Core Arguments")

        self.parser.add_argument(
            "--config", type=str, default=None, required=True, help="config yaml file"
        )
        self.parser.add_argument(
            "--config_override",
            type=str,
            default=None,
            help="Use to override config from command line directly",
        )
        self.parser.add_argument(
            "--local_rank",
            type=int,
            default=None,
            help="Local rank of the current node",
        )
        self.parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="Modify config options from command line, "
            "details can be seen from antmmf/common/build.py::build_config",
        )
        self.parser.add_argument(
            "--remote",
            dest="remote",
            action="store_true",
            help="Whether to perform tasks in the cloud."
            "If so, `cloud_attribute` must be given in "
            "configuration file.",
        )
        self.parser.add_argument(
            "--prj",
            default=None,
            help="The module name of your project, such as `mmbt`",
        )


flags = Flags()
