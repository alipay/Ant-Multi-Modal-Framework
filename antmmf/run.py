# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

"""
Build a trainer and launch it, you can wrap it in your own project.

Example::

    from antmmf.utils.env import setup_compatibility
    from antmmf.utils.flags import flags
    from antmmf.run import plain_run
    from mmbt import *     # noqa  make sure all modules have been registered.


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


For each project, you should create run.py in your own project to launch the trainer.
"""

import importlib
import traceback
from argparse import Namespace
from antmmf.common.build import build_config
from antmmf.trainers.build import build_trainer
from antmmf.utils.distributed_utils import is_main_process


def plain_run(args: Namespace):
    config = build_config(
        args.config,
        config_override=args.config_override,
        opts_override=args.opts,
        specific_override=args,
        args=args,
    )

    trainer = build_trainer(config, args)

    try:
        trainer.load()
        trainer.train()
    except Exception:
        writer = getattr(trainer, "writer", None)
        if writer is not None:
            # logging error to file
            writer.write(traceback.format_exc(), "error", donot_print=True)
        if is_main_process():
            # logging to stdout
            traceback.print_exc()
            raise
        exit(1)


def alps_run(entry_file=None):
    from alps.framework.engine import GpuConf
    from alps.framework.engine import ResourceConf
    from alps.pytorch.api.base.submit import submit_torch_task
    from alps.pytorch.api.runner.engine import TorchK8sEngine
    from alps.util import logger
    import os
    import sys

    args = sys.argv[1:]
    logger.info(f"args: {args}")
    if args.count("training_parameters.num_nodes") > 0:
        num_nodes = int(args[args.index("training_parameters.num_nodes") + 1])
        logger.info(f"========== 将使用{num_nodes}台机器进行训练")
    else:
        num_nodes = 1
        logger.info(f"========== 将使用单机进行训练")

    if args.count("training_parameters.num_gpus_per_node") > 0:
        num_gpus_per_node = int(
            args[args.index("training_parameters.num_gpus_per_node") + 1]
        )
        logger.info(
            f"========== 将使用总计{num_nodes} * {num_gpus_per_node} = {num_nodes * num_gpus_per_node}卡进行训练"
        )
    else:
        num_gpus_per_node = 0
        logger.info(f"========== 将使用CPU进行训练")

    if args.count("submit_parameters.num_cpus_per_node") > 0:
        arg_index = args.index("submit_parameters.num_cpus_per_node")
        num_cpus_per_node = int(args[arg_index + 1])
        args.pop(arg_index + 1)
        args.pop(arg_index)
    else:
        num_cpus_per_node = num_gpus_per_node * 8
    logger.info(f"========== 每台机器CPU数量: {num_cpus_per_node}")

    if num_cpus_per_node == 0:
        raise ValueError(
            f"参数training_parameters.num_gpus_per_node和submit_parameters.num_cpus_per_node必须设置一个"
        )

    if args.count("submit_parameters.memory") > 0:
        arg_index = args.index("submit_parameters.memory")
        memory = int(args[arg_index + 1]) * 1024
        args.pop(arg_index + 1)
        args.pop(arg_index)
    else:
        memory = num_gpus_per_node * 16 * 1024
    logger.info(f"========== 每台机器memory: {memory} MB")

    if args.count("submit_parameters.disk_quota") > 0:
        arg_index = args.index("submit_parameters.disk_quota")
        disk_quota = int(args[arg_index + 1]) * 1024
        args.pop(arg_index + 1)
        args.pop(arg_index)
    else:
        disk_quota = 40 * 1024
    logger.info(f"========== 每台机器硬盘: {disk_quota} MB")

    if args.count("submit_parameters.gpu_type") > 0:
        arg_index = args.index("submit_parameters.gpu_type")
        gpu_type = args[arg_index + 1]
        logger.info(f"========== GPU类型: {gpu_type}")
        args.pop(arg_index + 1)
        args.pop(arg_index)
    else:
        gpu_type = None

    if args.count("submit_parameters.app") > 0:
        arg_index = args.index("submit_parameters.app")
        app = args[arg_index + 1]
        priority = "high"
        args.pop(arg_index + 1)
        args.pop(arg_index)
    else:
        app = None
        priority = None
    logger.info(f"========== app name: {app}")

    if args.count("submit_parameters.priority") > 0:
        arg_index = args.index("submit_parameters.priority")
        priority = args[arg_index + 1]
        args.pop(arg_index + 1)
        args.pop(arg_index)
    logger.info(f"========== priority: {priority}")

    if args.count("submit_parameters.cluster") > 0:
        arg_index = args.index("submit_parameters.cluster")
        cluster = args[arg_index + 1]
        args.pop(arg_index + 1)
        args.pop(arg_index)
    else:
        cluster = None
    logger.info(f"========== cluster: {cluster}")

    if args.count("submit_parameters.user") > 0:
        arg_index = args.index("submit_parameters.user")
        user = args[arg_index + 1]
        args.pop(arg_index + 1)
        args.pop(arg_index)
    else:
        raise ValueError("任务提交需设置submit_parameters.user参数为用户工号")
    logger.info(f"========== 用户工号: {user}")

    # 使用alps trainer
    if args.count("training_parameters.trainer") == 0:
        args.append("training_parameters.trainer")
        args.append("alps_trainer")

    gpu = GpuConf(count=num_gpus_per_node, gpu_type=gpu_type)
    worker = ResourceConf(
        num=num_nodes,
        core=num_cpus_per_node,
        memory=memory,
        gpu=gpu,
        disk_quota=disk_quota,
    )
    k8s_engine = TorchK8sEngine(
        worker=worker,
        image="reg.docker.alibaba-inc.com/aii/aistudio:2179-20230324165747",
        app=app,
        priority=priority,
        cluster=cluster,
    )
    dir_path = os.path.dirname(sys.argv[0])
    if entry_file is None:
        entry_file = os.path.join(dir_path, "run.py")
    init_rc = os.path.join(os.path.dirname(entry_file), "init.rc")
    submit_torch_task(
        entry_file, k8s_engine, args=args, user=user, init_rc=init_rc, name="AntMMFTask"
    )


if __name__ == "__main__":
    from antmmf.utils.flags import flags

    parser = flags.get_parser()
    args = parser.parse_args()
    if str(args.prj) != "None":
        importlib.import_module(args.prj)
    plain_run(args)
