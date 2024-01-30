# Copyright (c) 2023 Ant Group and its affiliates.
import collections.abc
import glob
import os
import pickle
import sys
import warnings
from typing import AnyStr

import torch
from omegaconf import OmegaConf

from antmmf.common import constants
from antmmf.common.checkpoint import Checkpoint
from antmmf.common.registry import registry
from antmmf.common.report import Report, default_result_formater
from antmmf.models.build import build_model
from antmmf.structures import Sample, SampleList
from antmmf.utils.logger import StdoutLogger
from antmmf.utils.timer import Timer


class BasePredictor(object):
    def __init__(self, config):
        self.config = config
        self.predictor_parameters = self.config.predictor_parameters
        self.profiler = Timer()
        # for onnx inference session
        self.sess = None
        self.extra_data = None
        self.device = torch.device("cpu")

    def save_config(self, save_file):
        conf = OmegaConf.create(self.config.to_dict())
        OmegaConf.save(config=conf, f=save_file)

    def set_predictor_state(self):
        self._init_inference_device()

        # flag indicates online serving or local train/val/test state
        registry.register(constants.STATE, constants.STATE_ONLINE_SERVING)

        self.writer = StdoutLogger(self.config)
        registry.register(constants.WRITER_STR, self.writer)

        self.configuration = registry.get("configuration")
        self.writer.write(self.configuration)

    def load(self, with_ckpt=True, load_model=True):
        """

        :param with_ckpt: if load_model=True, load model weights if provided
        :param load_model: if False, only load processors
        :return:
        """
        self.set_predictor_state()
        # Some models need to be built with resource file configured first
        self.extra_data = self.load_extras()
        if load_model:
            self.load_model()
            if with_ckpt:  # support random initialization when unit test
                self.load_checkpoint()
        self.load_processors()

    def override_with_value(self, key_path: AnyStr, value: AnyStr) -> None:
        """
        Typically used for using predictor_config to update model config before
        it was built.

        Args:
            key_path(str): path to leaf node in yaml, separated with '.', eg.  'model_attributes.mmbt'
            value: update value

        Returns:

        """
        self.config.defrost()  # make config writable
        node = self.config
        ful_path = key_path.split(".")
        node_path, value_name = ful_path[:-1], ful_path[-1]
        for path in node_path:
            node = getattr(node, path, None)
            if node is None:
                break
        if node is None:
            warnings.warn(
                f"Failed to override config node:{key_path} with value:{value}"
            )
        else:
            assert isinstance(node, collections.abc.Mapping)
            setattr(node, value_name, value)
        self.config.freeze()

    def _init_inference_device(self):
        # Default device is cpu
        self.device = self.predictor_parameters.device
        self.local_rank = self.predictor_parameters.get("local_rank", 0)
        if torch.cuda.is_available():
            # Collie will always use env variable `CUDA_VISIBLE_DEVICES` to control services' accessment to GPUs.
            # During inference, try to use cuda whenever available
            torch.cuda.set_device(self.local_rank)
            self.device = "cuda:0"
        registry.register("current_device", self.device)

    def load_processors(self):
        processor_config = self.predictor_parameters.get("processors", None)
        if processor_config is None:
            self.processors = None
            return
        from antmmf.datasets.build import build_processors

        self.processors = build_processors(processor_config)

    def load_model(self):
        assert (
            len(self.config.model_attributes) == 1
        ), "There should be only one model in model_attributes"
        model_key = list(self.config.model_attributes.keys())[0]
        attributes = self.config.model_attributes[model_key]

        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_attributes[attributes]

        # make config be writable
        self.config.defrost()

        attributes["model"] = model_key
        # models should implement build_for_test for online serving
        self.model = build_model(attributes, for_test=True)

        self.config.freeze()

        training_parameters = self.config.training_parameters

        data_parallel = training_parameters.data_parallel
        distributed = training_parameters.distributed

        registry.register("data_parallel", data_parallel)
        registry.register("distributed", distributed)

        if "cuda" in str(self.config.predictor_parameters.device):
            rank = self.local_rank if self.local_rank is not None else 0
            device_info = "CUDA Device {} is: {}".format(
                rank, torch.cuda.get_device_name(self.local_rank)
            )

            self.writer.write(device_info, log_all=True)

        self.model = self.model.to(self.device)
        self.model.eval()  # for inference only

        self.writer.write("Torch version is: " + torch.__version__)

    def load_checkpoint(self):
        # load checkpoint saved by antmmf
        self.resume_file = None
        ckpt_ext = ["*.pth", "*.ckpt"]
        for ext in ckpt_ext:
            files = glob.glob(
                os.path.join(self.config.predictor_parameters.model_dir, ext)
            )  # return list
            if len(files) == 1:
                self.resume_file = files[0]
                break
            if len(files) > 1:
                raise ValueError(
                    f"Multiple checkpoints found in directory:{self.config.predictor_parameters.model_dir}"
                )
        if self.resume_file is None:
            raise ValueError(
                f"No checkpoints (*.pth, *.ckpt) found in directory: {self.config.predictor_parameters.model_dir}"
            )
        self.checkpoint = Checkpoint(self, load_only=True)
        self.checkpoint.load_model_weights(self.resume_file, force=True)

    def _forward_pass(self, samplelist):
        # Arguments should be a samplist at this point
        model_output = self.model(samplelist)
        report = Report(samplelist, model_output)
        self.profile("Forward time")
        return report

    def set_resources(self, resource_dict={}):
        """
        handling predictors' dependency resource paths, paths are probably passed from collie.
        Args:
            resource_dict: {resource_name: resource_path}

        Returns:

        """
        for k, v in resource_dict.items():
            setattr(self, k, v)

    def load_extras(self):
        """
        load any inference related model or resources
        """
        return None

    def dummy_request(self):
        """
        mock online request_input for local test
        Returns:
            input to _build_sample

        """
        raise NotImplementedError

    def _build_sample(self, data, json_obj):
        """

        Args:
            data: image bytes
            json_obj: one json object include multimodal info

        Returns:
            sample(antmmf.structures.sample.Sample)

        """
        raise NotImplementedError

    def _predict(self, sample_list):
        with torch.no_grad():
            sample_list = sample_list.to(self.device)
            report = self._forward_pass(sample_list)
        return report

    def predict(self, data=None, json_obj=None):
        """
        Args:
            data: image bytes
            json_obj: one json object include multimodal info

        Returns:
            Report(OrderedDict)
        """
        if data is None:
            data, json_obj = self.dummy_request()
        self.profiler.reset()
        sample = self._build_sample(data, json_obj)
        self.profile("Build sample time")
        if not isinstance(sample, Sample):
            raise Exception(
                f"Method _build_sample is expected to return a instance of antmmf.structures.sample.Sample,"
                f"but got type {type(sample)} instead."
            )
        result = self._predict(SampleList([sample]))
        np_result = default_result_formater(result)
        result = self.format_result(np_result)
        assert isinstance(
            result, dict
        ), f"Result should be instance of Dict,but got f{type(result)} instead"
        return result

    def format_result(self, report):
        """
        online returned format
        Args:
            report(antmmf.common.report.Report):

        Returns:
            format_result(dict): must be serializable by `json.dumps` for online serving

        """
        print(f"report:{report}")
        return report

    def profile(self, text):
        sys.stdout.write(text + ": " + self.profiler.get_time_since_start() + "\n")
        self.profiler.reset()

    def __getstate__(self):
        states = {
            "config": self.config,
            "processors": self.processors,
            "extra_data": self.extra_data,
        }
        if hasattr(self, "model"):
            states.update({"model": self.model})
        return pickle.dumps(states)

    def __setstate__(self, state):
        state_dict = pickle.loads(state)
        self.__init__(state_dict["config"])
        self.processors = state_dict["processors"]
        self.extra_data = state_dict["extra_data"]
        if "model" in state_dict:
            self.model = state_dict["model"]
        self.set_predictor_state()
        if "model" in state_dict:
            self.model = self.model.to(self.device)
            self.model.eval()
