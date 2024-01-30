# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import json
import os
import re
import requests
import yaml
import inspect
import collections.abc
from dataclasses import asdict
from typing import Union, Dict, List, Tuple, Any, Iterable, Callable, KeysView
from omegaconf import OmegaConf, ListConfig
from ast import literal_eval

from transformers.configuration_utils import PretrainedConfig
from antmmf.utils.general import get_antmmf_root
from antmmf.common.registry import registry


def string_to_bool(string_val: Union[str, bool]):
    if not isinstance(string_val, bool):
        string_val = str(string_val)
        if string_val.lower() == "true":
            string_val = True
        elif string_val.lower() == "false":
            string_val = False
        else:
            raise NotImplementedError(
                f"string_val must be bool or a bool like string, but got {string_val}"
            )
    return string_val


# ******   Make Configuration can be json serializable.   ******
def default(self, obj):
    if isinstance(obj, Configuration):
        return obj.to_dict()
    return json.JSONEncoder.default(self, obj)


json.JSONEncoder.default = default
# ****** Make Configuration can be json serializable. end ******


def _decode_value(value):
    # https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L400
    if not isinstance(value, str):
        return value

    if value == "None":
        value = None

    try:
        value = literal_eval(value)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return value


def nested_dict_update(dictionary, update):
    """Updates a dictionary with other dictionary recursively.

    Parameters
    ----------
    dictionary : dict
        Dictionary to be updated.
    update : dict
        Dictionary which has to be added to original one.

    Returns
    -------
    dict
        Updated dictionary.
    """
    if dictionary is None:
        dictionary = {}

    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            dictionary[k] = nested_dict_update(dictionary.get(k, {}), v)
        else:
            dictionary[k] = _decode_value(v)
    return dictionary


def convert_value_to_str(value: Any):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, List):
        return [convert_value_to_str(item) for item in value]
    if isinstance(value, Tuple):
        return (convert_value_to_str(item) for item in value)
    if isinstance(value, collections.abc.Mapping):
        return {k: convert_value_to_str(v) for k, v in value.items()}
    if isinstance(value, Iterable):
        # numpy, torch, etc.
        if hasattr(value, "tolist") and isinstance(value.tolist, Callable):
            value = value.tolist()
        return [convert_value_to_str(val) for val in value]
    return value


def load_from_file(file_path: str) -> Dict:
    # support http
    if file_path.startswith("http"):
        package = requests.get(file_path, verify=False)
        if str(package.status_code) == "200":
            config = os.path.expandvars(package.content)
            config = yaml.load(config, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError(f"No such file: {file_path}")
    else:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No such file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as stream:
            # replace $env_var or ${env_var} in yaml file
            # see detail at: https://stackoverflow.com/questions/52412297/\
            # how-to-replace-environment-variable-value-in-yaml-file-to-be-parsed-using-python
            stream = os.path.expandvars(stream.read())
            config = yaml.load(stream, Loader=yaml.FullLoader)

    include_config = {}
    for path in config.get("includes", []):
        # make compatible with old codes
        if os.path.isfile(os.path.join(get_antmmf_root(), path)):
            path = os.path.join(get_antmmf_root(), path)
        elif not os.path.isabs(path):
            # new feature: using relative path with the dir of file_path as root dir
            path = file_path.split("/")[:-1] + path.split("/")
            path = os.path.join(*path)

        nested_dict_update(include_config, load_from_file(path))

    nested_dict_update(include_config, config)
    include_config.pop("includes", None)
    return include_config


def wrap_to_configuration(val):
    val = _decode_value(val)
    if isinstance(val, collections.abc.Mapping):
        return Configuration(val)
    if isinstance(val, (List, Tuple, ListConfig)):
        return [wrap_to_configuration(v) for v in val]
    return val


def parse_field(fields: str) -> List[str]:
    """
    Usage:
        transform the given string fields to List[field], for example:
            1. given: "metrics.[0].type"    return: ["metrics", "[0]", "type"]
            2. given: "metrics[0].type"    return: ["metrics", "[0]", "type"]
    """
    fields = fields.split(".")
    ret_fields = []
    for field in fields:
        # process the field like: "[0]"
        if re.match(r"\[(\d+)\]", field) is not None:
            ret_fields.append(field)
            continue
        # process the field like: "metrics[0]"
        sub_field = re.sub(r"(\[\d+\])+$", "", field)
        if sub_field != field:
            indices = re.findall(r"(\[\d+\])", field.replace(sub_field, ""))
            if sub_field != "":
                ret_fields.append(sub_field)
            ret_fields.extend(indices)
            continue
        ret_fields.append(field)
    return ret_fields


def get_zoo_config(
    key, variation="defaults", zoo_config_path=None, zoo_type="datasets"
):
    version = None
    resources = None
    if zoo_config_path is None:
        zoo_config_path = os.path.join("configs", "zoo", f"{zoo_type}.yaml")
    if os.path.exists(zoo_config_path):
        zoo = yaml.safe_load(open(zoo_config_path))
        if key not in zoo:
            return version, resources
        zoo = zoo[key]
        zoo = zoo[variation]
        version = zoo["version"]
        resources = zoo["resources"]

    return version, resources


class AntMMFConfig(object):
    """
    AntMMF config is used to filter data fields from a global config,
    such making it clear to identity fields that are indeed necessary.
    It is inspired from:
    https://stackoverflow.com/questions/54678337/how-does-one-ignore-extra-arguments-passed-to-a-data-class
    """

    # data fields
    # eg. example_field: int = 1

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def create_from(cls, config: Union[Dict, PretrainedConfig] = None, **kwargs):
        if isinstance(config, (cls, PretrainedConfig)):
            # AntMMFConfig/transformer config -> dict
            config = config.to_dict()
        # update config with kwargs
        if hasattr(config, "defrost"):
            config.defrost()
        config = nested_dict_update(config, kwargs)
        if hasattr(config, "freeze"):
            config.freeze()

        # rebuild AntMMFConfig from updated config
        return cls.from_dict(config)

    def get(self, key, default_val=None):
        # keep compatibility with dict style config
        return getattr(self, key, default_val)

    def __getitem__(self, key):
        return self.get(key)


class Configuration(collections.abc.Mapping):
    """
    Args:
        config: path of the yaml file or dict of the configuration
    Usage:

    .. code-block:: python

        # construct from any collections.abc.Mapping (including Dict, DictConfig, etc.)
        cfg = Configuration({"var": 0, "var1": "1,2,4,3", "other": {"var2": "conf", "size": 4}})
        assert cfg.var == 0 and cfg.var1 == (1,2,4,3) and cfg.var2 == "conf"
        dict_config = OmegaConf.create("your/yaml/or/json/file")
        cfg = Configuration(dict_config)
        print(cfg)

        # directly load configuration from yaml or json file
        cfg = Configuration("your/yaml/or/json/file")

        # load the default configuration
        cfg.set_default()
        print(cfg)

        # freeze or defrost your config
        cfg.freeze() # or cfg = cfg.freeze()
        cfg.defrost() # or cfg = cfg.defrost()

        # merge two configuration
        cfg1 = Configuration()
        cfg.update(cfg1)
    """

    def __init__(
        self,
        config: Union[str, collections.abc.Mapping] = None,
    ):

        self.__dict__["_content"] = {}
        self.__dict__["_flags_cache"] = {}

        if config is None:
            self.with_default()
            return

        if isinstance(config, Configuration):
            for key, value in config.items():
                self[key] = value
            return

        if isinstance(config, str):
            # load configuration from yaml file
            config = load_from_file(config)
            self.from_dict_conf(config)
            self.with_default()
        elif isinstance(config, collections.abc.Mapping):
            self.from_dict_conf(config)
        else:
            raise NotImplementedError(
                f"The type of {type(config)} is not support in Configuration"
            )

    def with_default(self) -> "Configuration":
        """
        Load default configuration from antmmf/common/defaults/configs/base.yaml if the configuration is not exist,
        and the configurations you specified before call this function will not be changed.
        """
        directory = os.path.dirname(os.path.abspath(__file__))
        default_file_path = os.path.join(directory, "defaults", "configs", "base.yml")

        config = OmegaConf.load(default_file_path)
        config = Configuration(config)
        config.update(self)
        self.from_dict_conf(config)
        return self

    def override_with_cmd_config(
        self, cmd_config: Union[str, collections.abc.Mapping]
    ) -> "Configuration":
        """
        Override the configurations, it is usually used in command line.

        Example:

        .. code-block:: python

            # Usage 1: override configurations from YAML file
            base_config: Configuration = ...
            cmd_config = "your override configuration file.yml"
            base_config = base_config.override_with_cmd_config(cmd_file)
            # Usage 2: override configurations from any mapping
            base_config: Configuration = ...
            cmd_config: collections.abc.Mapping = ...
            base_config = base_config.override_with_cmd_config(cmd_file)

        Args:
            cmd_config (str or collections.abc.Mapping): the YAML file path of configuration or any mapping.

        Returns:
            Configuration: the configuration itself
        """
        if cmd_config is not None:
            config = Configuration(cmd_config)
            self.update(config)

        return self

    def override_with_cmd_opts(self, opts: List) -> "Configuration":
        """
        Merge options to configurations.

        Example:

        .. code-block:: python

            config = Configuration({"a": 1, "b": {"c": 2, "d": 3}})
            opts = ["a", 4, "b.d", 5]
            config = config.override_with_cmd_opts(opts)
            print(config.b.d)   # 5

        Args:
            opts (list): a list of [key1, value1, key2, value2, ...]

        Returns:
            Configuration: the configuration itself
        """
        self._merge_from_list(opts)
        return self

    def update(self, others: collections.abc.Mapping) -> "Configuration":
        """
        Update the config in an inplace way.

        Usage:

        .. code-block:: python

            config: Configuration = ...
            update_config: collections.abc.Mapping = ...
            config = config.update(cmd_file)

        Args:
            others (collections.abc.Mapping): any mapping which contains configurations.

        Returns:
            Configuration: the configuration itself
        """
        conf1 = self.to_dict()
        if isinstance(others, Configuration):
            others = others.to_dict()
        nested_dict_update(conf1, others)
        self.from_dict_conf(conf1)
        return self

    def merge_with(self, others: Union[collections.abc.Mapping]) -> "Configuration":
        """
        Make it compatible to the old Configuration, and its function is same as update.

        Returns:
            Configuration: the configuration itself
        """
        return self.update(others)

    def _merge_from_list(self, opts) -> "Configuration":
        if opts is None:
            opts = []

        assert (
            len(opts) % 2 == 0
        ), "Number of opts should be multiple of 2, but got {}, opts is {}".format(
            len(opts), opts
        )

        for opt, value in zip(opts[0::2], opts[1::2]):
            splits = parse_field(opt)
            current = self
            for idx, field in enumerate(splits):
                # support Sequence index, eg.
                # transforms:
                #    - type: Resize
                #      params:
                #          size: [256, 256]
                #    - type: RandomCrop
                #      params:
                #          size: [224, 224]
                # override size [224, 224] -> [300, 300] with the following literal:
                # transforms.[1].params.size [300,300]

                # TODO: if the key is `transforms.[1]` like, then the value will not be updated
                seq_idx = re.match(r"\[(\d+)\]", field)
                if isinstance(current, (List, Tuple)) and seq_idx:
                    if idx == len(splits) - 1:
                        current[int(seq_idx.group(1))] = wrap_to_configuration(value)
                    else:
                        current = current[int(seq_idx.group(1))]
                    continue

                if (
                    isinstance(current, collections.abc.Mapping)
                    and field not in current
                ):
                    raise AttributeError(
                        f"While updating configuration option {opt} is missing from"
                        f" configuration {current} at field {field}\n"
                        f" perhaps remove -- from the parameter if there is -- before the argument"
                    )

                if not isinstance(current[field], collections.abc.Mapping):
                    if idx == len(splits) - 1:  # indicate leaf node
                        current[field] = wrap_to_configuration(value)
                    else:  # intermediate nodes that are not Mapping must be Sequence.
                        if isinstance(current[field], collections.abc.Sequence):
                            current = current[field]
                        else:
                            raise AttributeError(
                                "While updating configuration",
                                "option {} is not present after field {}".format(
                                    opt, field
                                ),
                            )
                else:
                    current = current[field]

        return self

    def freeze(self) -> "Configuration":
        """
        Freeze the configuration, and all nodes in any depth can not be editable.
        """
        for field, value in self.__dict__["_content"].items():
            if isinstance(value, Configuration):
                value.freeze()
            elif isinstance(value, (List, Tuple)):
                for item in value:
                    if isinstance(item, Configuration):
                        item.freeze()

        self.__dict__["_flags_cache"]["readonly"] = True
        return self

    def defrost(self) -> "Configuration":
        """
        Defrost the configuration, make it can be editable.
        """
        for field, value in self.__dict__["_content"].items():
            if isinstance(value, Configuration):
                value.defrost()
            elif isinstance(value, (List, Tuple)):
                for item in value:
                    if isinstance(item, Configuration):
                        item.defrost()

        self.__dict__["_flags_cache"]["readonly"] = False
        return self

    def is_frozen(self) -> bool:
        """
        Whether the configuration is frozen.
        """
        return self.__dict__["_flags_cache"].get("readonly", False)

    def to_dict(self) -> Dict:
        """
        Convert the configuration to dictionary.
        """
        ret = {}
        for key, value in self.__dict__["_content"].items():
            if isinstance(value, Configuration):
                value = value.to_dict()
            if isinstance(value, (List, Tuple)):
                value = [
                    v.to_dict() if isinstance(v, Configuration) else v for v in value
                ]
            ret[key] = value
        return ret

    def _convert_value_to_str(self):
        ret = {}
        for key, value in self.__dict__["_content"].items():
            if isinstance(value, Configuration):
                value = value._convert_value_to_str()
            else:
                value = convert_value_to_str(value)
            ret[key] = value
        return ret

    def from_dict_conf(self, conf: collections.abc.Mapping) -> "Configuration":
        """
        Construct configuration from any mapping.
        """
        for key, value in conf.items():
            self.__setattr__(key, value)
        return self

    def keys(self) -> KeysView:
        """
        Returns:
            All keys of this configuration with depth of 1.
        """
        return self.__dict__["_content"].keys()

    def items(self) -> List:
        """
        Returns:
            A list of (key, value) pair, and the function is similar to `dict`
        """
        ret = []
        for key, value in self.__dict__["_content"].items():
            ret.append((key, value))
        return ret

    def get(self, key, default_value=None) -> Any:
        """
        Returns:
            The value of the key, if the key not exist, it will return a default value.
        """
        return self[key] if key in self else default_value

    def pop(self, key, default_value=None) -> Any:
        """
        Remove the node of the key, and return its value, its function is similar to the `get` of dict.
        """
        if self.is_frozen():
            RuntimeError("Configuration has been frozen and can't be edited.")
        return self.__dict__["_content"].pop(key, default_value)

    def __len__(self):
        return len(self.__dict__["_content"])

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __setstate__(self, state: Dict):
        self.__dict__["_content"] = state.get("_content", {})
        self.__dict__["_flags_cache"] = state.get("_flags_cache", {})

    def __getstate__(self) -> Dict:
        return self.__dict__

    def __setattr__(self, key, value):
        if self.is_frozen():
            raise AttributeError("Configuration has been frozen and can't be updated")

        self.__dict__["_content"][key] = wrap_to_configuration(value)

    def __getattr__(self, key):
        if key not in self.__dict__["_content"]:
            raise AttributeError(f"The key or function of {key} is not found.")

        return self.__dict__["_content"][key]

    def __delitem__(self, key):
        """
        usage:

        .. code-block:: python

            conf = Configuration("you/config/file")
            del conf["training_parameters"]
        """
        if self.is_frozen():
            raise AttributeError("Configuration has been frozen and can't be delete")
        del self.__dict__["_content"][key]

    def __del__(self):
        del self.__dict__["_content"]
        del self.__dict__["_flags_cache"]

    def __contains__(self, key):
        """
        override `in` operator

        usage:

        .. code-block:: python
            dict_conf = {"var": 0, "var1": "1,2,4,3", "other": {"var2": "conf", "size": 4}}
            cfg = Configuration(dict_conf)
            assert "var" in cfg
            assert "var2" not in cfg
        """
        return key in self.__dict__["_content"]

    def __iter__(self):
        """
        usage:

        .. code-block:: python
            conf = Configuration("you/config/file")
            for k in conf:
                value = conf[k]
        """

        for key in self.keys():
            yield key

    def __str__(self):
        config = self._convert_value_to_str()
        return yaml.dump(config)

    def __repr__(self):
        config = self._convert_value_to_str()
        return yaml.dump(config)


def pretty_print(config: Configuration):
    """
    Print the configuration in a pretty way.

    Args:
        config (Configuration): which must include the default configurations
        which shown in antmmf/common/defaults/configs/base.yaml
    """
    writer = registry.get("writer")

    writer.write("=====  Training Parameters    =====", "info")
    writer.write(
        json.dumps(config.training_parameters, indent=4, sort_keys=True),
        "info",
    )

    writer.write("======  Task Attributes  ======", "info")

    for task in config.task_attributes:

        task_config = config.task_attributes[task]

        for dataset in task_config.dataset_attributes:
            writer.write("======== {}/{} =======".format(task, dataset), "info")
            dataset_config = task_config.dataset_attributes[dataset]
            writer.write(json.dumps(dataset_config, indent=4, sort_keys=True), "info")

    writer.write("======  Optimizer Attributes  ======", "info")
    writer.write(
        json.dumps(config.optimizer_attributes, indent=4, sort_keys=True),
        "info",
    )

    model_name = "-".join(config.model_attributes.keys())
    writer.write("======  Model ({}) Attributes  ======".format(model_name), "info")
    writer.write(
        json.dumps(
            config.model_attributes,
            indent=4,
            sort_keys=True,
        ),
        "info",
    )
