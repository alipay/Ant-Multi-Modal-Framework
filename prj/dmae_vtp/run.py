# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.utils.env import setup_compatibility
from antmmf.utils.flags import flags
from antmmf.run import plain_run

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from roi_univl import *  # noqa  make sure all modules have been registered.


def run():
    parser = flags.get_parser()
    try:
        args = parser.parse_args()
        plain_run(args)
    except SystemExit:
        exit(2)


if __name__ == "__main__":
    setup_compatibility()
    run()
