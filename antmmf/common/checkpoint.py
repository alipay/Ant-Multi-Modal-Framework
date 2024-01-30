# Copyright (c) 2023 Ant Group and its affiliates.
import glob
import os
import warnings

import torch

from antmmf.common import constants, Configuration
from antmmf.common.registry import registry
from antmmf.utils.distributed_utils import is_main_process, synchronize
from antmmf.utils.general import (
    ckpt_name_from_core_args,
    updir,
)


def load_state_dict_mapping(model, ckpt_model, attr_mapping):
    """
    load checkpoint and then map the key of the loaded checkpoint
    to a key in a target model

    Example of attr_mapping:
        attr_mapping = {
            "image_feature_encoders": "img_feat_encoders",
            "image_feature_embeddings_list": "img_embeddings_list",
            "image_text_multi_modal_combine_layer": "multi_modal_combine_layer",
            "text_embeddings": "text_embeddings",
            "classifier": "classifier",
        }

    An example of a configuration using this function is at
    configs/iot/adversarial/iot.local.yml
        pretrained:
        - softmax_w:
            source_key: weight
            source_model: ../../data/iot/models/zs_with_da_iter_380000.pth_0

    Attribute:
    model: target model
    ckpt_model: source model
    attr_mapping: the mapping of the target key to source key

    """

    own_state = model.state_dict()
    for key in attr_mapping:
        own_state[key].copy_(ckpt_model[attr_mapping[key]])


def load_pretrained_model(model_name, *args, **kwargs):
    antmmf_pretrained_cache = os.environ.get(
        constants.ANTMMF_PRETRAINED_MODELS_ENV_VAR, ""
    )
    model_key = model_name.split(".")[0]
    model_path = os.path.join(antmmf_pretrained_cache, model_key, model_name)
    if os.path.exists(model_path):
        download_path = model_path

    # get config file if exists
    configs = []
    allowed_config_types = ["*.yml", "*.yaml"]
    for config_type in allowed_config_types:
        configs.extend(glob.glob(os.path.join(download_path, config_type)))
    assert len(configs) == 1, "Only accept one config file with the pretrained model."
    # 这里model单指模型文件本身，不包含config等其它信息, 后缀为pth
    models = []
    models.extend(glob.glob(os.path.join(download_path, "*.pth")))
    assert len(models) == 1, "None or multiple checkpoint files."
    model = torch.load(models[0], map_location=lambda storage, loc: storage)
    config = Configuration(configs[0])

    model_attr = config.get("model_attributes")
    assert len(model_attr) == 1, "Model number should be 1"
    model_key = list(model_attr.keys())[0]
    model_config = model_attr[model_key]

    return {"config": model_config, "model": model, "full_config": config}


class Checkpoint:
    def __init__(self, trainer, load_only=False):
        """
        Generates a path for saving model which can also be used for resuming
        from a checkpoint.
        """
        self.trainer = trainer

        self.config = self.trainer.config
        self.save_dir = self.config.training_parameters.save_dir
        self.model_name = "-".join(self.config.model_attributes.keys())

        self.ckpt_foldername = ckpt_name_from_core_args(self.config)

        # There is no Compatibility issues coming up for now.
        # if getattr(self.trainer, "args", None) is not None:
        #     self.ckpt_foldername += foldername_from_config_override(
        #         self.trainer.args)

        self.device = registry.get("current_device")

        self.ckpt_prefix = ""

        if hasattr(self.trainer.model, "get_ckpt_name"):
            self.ckpt_prefix = self.trainer.model.get_ckpt_name() + "_"

        self.config.defrost()
        self.config["log_foldername"] = self.ckpt_foldername
        self.config.freeze()

        self.ckpt_foldername = os.path.join(self.save_dir, self.ckpt_foldername)
        self.pth_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + self.model_name + "_final.pth"
        )

        self.models_foldername = os.path.join(self.ckpt_foldername, "models")

        if not load_only:  # disable mkdir & write config when load_only=True
            if not os.path.exists(self.models_foldername):
                os.makedirs(self.models_foldername, exist_ok=True)
            self.save_config()

        self.repo_path = updir(os.path.abspath(__file__), n=3)
        self.repo = None
        try:
            import git

            self.repo = git.Repo(self.repo_path)
        except Exception:
            print("no git repository")

        self.max_ckpt_num = self.config.training_parameters.max_ckpt_num

    def save_config(self):
        cfg_file = os.path.join(self.ckpt_foldername, "config.yaml")
        with open(cfg_file, "w") as f:
            # Pop out config_override if present to remove clutter in
            # saved configuration yaml file
            self.config.pop("config_override", None)
            f.write(str(self.config))

    def load_state_dict(self):
        tp = self.config.training_parameters
        if tp.resume_file is not None:
            if os.path.exists(tp.resume_file):
                self._load(tp.resume_file, resume_state=not tp.restart)
                return
            else:
                raise RuntimeError("{} doesn't exist".format(tp.resume_file))

        ckpt_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + "best.ckpt"
        )

        if tp.resume is True:
            if os.path.exists(ckpt_filepath):
                self._load(ckpt_filepath)
            else:
                warnings.warn(
                    "Tried to resume but checkpoint filepath {} is not present. Skipping.".format(
                        ckpt_filepath
                    )
                )

    def load_model_weights(self, file, force=False):
        self.trainer.writer.write("Loading checkpoint")
        ckpt = self._torch_load(file)

        # for online serving,  models will never be wrapped by
        # torch.nn.DataParallel or torch.nn.parallel.DistributedDataParallel
        if registry.get(constants.STATE) is constants.STATE_ONLINE_SERVING:
            data_parallel = False
        else:
            data_parallel = registry.get("data_parallel") or registry.get("distributed")

        if "model" in ckpt:
            ckpt_model = ckpt["model"]
        else:
            ckpt_model = ckpt
            ckpt = {"model": ckpt}

        new_dict = {}

        # TODO: Move to separate function
        for attr in ckpt_model:
            if "fa_history" in attr:
                new_dict[attr.replace("fa_history", "fa_context")] = ckpt_model[attr]
            elif data_parallel is False and attr.startswith("module."):
                # In case the ckpt was actually a data parallel model
                # replace first module. from dataparallel with empty string
                new_dict[attr.replace("module.", "", 1)] = ckpt_model[attr]
            elif data_parallel is not False and not attr.startswith("module."):
                new_dict["module." + attr] = ckpt_model[attr]
            else:
                new_dict[attr] = ckpt_model[attr]

        self._load_state_dict(new_dict)
        self._load_model_weights_with_mapping(new_dict, force=force)
        return ckpt

    def _load_state_dict(self, state_dict):
        own_state = self.trainer.model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                warnings.warn(
                    f"loading checkpoint warning: skip loading tensor:{name} in checkpoint, which does not "
                    f"exist in model"
                )
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_tensor = own_state[name]
            if own_tensor.shape != param.shape:
                warnings.warn(
                    f"loading checkpoint warning: skip loading tensor:{name} in checkpoint, whose shape does "
                    f"not match model's tensor:{name}"
                )
                continue
            own_state[name].copy_(param)

    def _load_model_weights_with_mapping(self, weight_dict, force):
        pretrained_mapping = self.config.training_parameters.pretrained_mapping
        if not self.config.training_parameters.load_pretrained or force is True:
            pretrained_mapping = {}

        if len(pretrained_mapping.items()) == 0:
            return

        model = self.trainer.model
        own_state = model.state_dict()

        for key, value in pretrained_mapping.items():
            key += "."
            value += "."
            for attr in weight_dict:
                for own_attr in own_state:
                    if (
                        key in attr
                        and value in own_attr
                        and attr.replace(key, "") == own_attr.replace(value, "")
                    ):
                        self.trainer.writer.write("Copying " + attr + " " + own_attr)
                        own_state[own_attr].copy_(weight_dict[attr])
        self.trainer.writer.write("Pretrained model loaded")

    def _load(self, file, force=False, resume_state=False):
        ckpt = self.load_model_weights(file, force=force)

        # skip loading training state
        if resume_state is False:
            return

        if "optimizer" in ckpt:
            self.trainer.optimizer.load_state_dict(ckpt["optimizer"])
            # fix the bug of checkpoint in the pytorch with version higher than 1.11
            if "capturable" in self.trainer.optimizer.param_groups[0]:
                self.trainer.optimizer.param_groups[0]["capturable"] = True
        else:
            warnings.warn(
                "'optimizer' key is not present in the checkpoint asked to be loaded. Skipping."
            )

        self.trainer.early_stopping.init_from_checkpoint(ckpt)

        self.trainer.writer.write("Checkpoint {} loaded".format(file))

        if "current_iteration" in ckpt:
            self.trainer.current_iteration = ckpt["current_iteration"]
            registry.register("current_iteration", self.trainer.current_iteration)

        if "current_epoch" in ckpt:
            self.trainer.current_epoch = ckpt["current_epoch"]
            registry.register("current_epoch", self.trainer.current_epoch)

    def _torch_load(self, file):
        if "cuda" in str(self.device):
            self.trainer.writer.write(f"load model to {self.device}")
            return torch.load(file, map_location=self.device)
        else:
            return torch.load(file, map_location=lambda storage, loc: storage)

    def _get_vcs_fields(self):
        """Returns a dict with git fields of the current repository

        To reproduce an experiment directly from a checkpoint

        1) Export `config` key as a yaml
        2) Clone repository and checkout at given commit on given branch
        3) Any local change (diff) while running the experiment is stored
           in the value with key `git/diff`, output the diff to a `path.diff`
           file and apply the patch to the current state by simply

                        `patch -p0 < path.diff`
        """

        ret = {}
        if self.repo is not None:
            try:
                ret = {
                    "git/branch": self.repo.active_branch.name,
                    "git/commit_hash": self.repo.head.commit.name_rev,
                    "git/commit_author": self.repo.head.commit.author.name,
                    "git/commit_message": self.repo.head.commit.message,
                    "git/diff": self.repo.git.diff("--no-prefix"),
                }
            except Exception:
                pass
        return ret

    def remove_redundant_ckpts(self):
        ckpt_pattern = os.path.join(self.models_foldername, "model_*.ckpt")
        ckpt_lst = glob.glob(ckpt_pattern)
        if self.max_ckpt_num is not None and len(ckpt_lst) > self.max_ckpt_num:
            # sort checkpoints by modified time
            ckpt_lst = sorted(ckpt_lst, key=lambda x: os.path.getmtime(x))
            # remove redundant checkpoints
            for ckpt in ckpt_lst[: len(ckpt_lst) - self.max_ckpt_num]:
                os.remove(ckpt)

    def save(self, iteration, update_best=False):
        # Only save in main process
        if not is_main_process():
            return

        ckpt_filepath = os.path.join(
            self.models_foldername, "model_%d.ckpt" % iteration
        )
        best_ckpt_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + "best.ckpt"
        )

        best_iteration = self.trainer.early_stopping.best_monitored_iteration
        best_metric = self.trainer.early_stopping.best_monitored_value
        current_iteration = self.trainer.current_iteration
        current_epoch = self.trainer.current_epoch
        model = self.trainer.model
        data_parallel = registry.get("data_parallel") or registry.get("distributed")

        if data_parallel is True:
            model = model.module

        ckpt = {
            "model": model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "current_iteration": current_iteration,
            "current_epoch": current_epoch,
            "best_iteration": best_iteration,
            "best_metric_value": best_metric,
            # do not save config in ckpt, for it has already been saved
        }

        torch.save(ckpt, ckpt_filepath)
        self.remove_redundant_ckpts()

        if update_best:
            torch.save(ckpt, best_ckpt_filepath)

    def restore(self, with_sync=True):
        # make sure all processes load same model
        if with_sync:
            synchronize()
        self.trainer.writer.write("Restoring checkpoint")
        best_path = os.path.join(self.ckpt_foldername, self.ckpt_prefix + "best.ckpt")

        if os.path.exists(best_path):
            self._load(best_path, force=True)

    def finalize(self):
        torch.save(self.trainer.model.state_dict(), self.pth_filepath)
