# Copyright (c) 2023 Ant Group and its affiliates.
"""
The processors exist in antmmf to make data processing pipelines in various
datasets as similar as possible while allowing code reuse.

The processors also help maintain proper abstractions to keep only what matters
inside the dataset's code. This allows us to keep the dataset ``get_item``
logic really clean and no need about maintaining opinions about data type.
Processors can work on both images and text due to their generic structure.

To create a new processor, follow these steps:

1. Inherit the ``BaseProcessor`` class.
2. Implement ``_call`` function which takes in a dict and returns a dict with
   same keys preprocessed as well as any extra keys that need to be returned.
3. Register the processor using ``@registry.register_processor('name')`` to
   registry where 'name' will be used to refer to your processor later.

In processor's config you can specify ``preprocessor`` option to specify
different kind of preprocessors you want in your dataset.

Let's break down processor's config inside a dataset (VQA2.0) a bit to understand
different moving parts.

Config::

    task_attributes:
        vqa:
            dataset_attributes:
                vqa2:
                    processors:
                      text_processor:
                        type: vocab
                        params:
                          max_length: 14
                          vocab:
                            type: intersected
                            embedding_name: glove.6B.300d
                            vocab_file: vocabs/vocabulary_100k.txt
                          answer_processor:
                            type: vqa_answer
                            params:
                              num_answers: 10
                              vocab_file: vocabs/answers_vqa.txt
                              preprocessor:
                                type: simple_word
                                params: {}

``BaseDataset`` will init the processors and they will be available inside your
dataset with same attribute name as the key name, e.g. `text_processor` will
be available as `self.text_processor` inside your dataset. As is with every module
in antmmf, processor also accept a ``Configuration`` with a `type` and `params`
attributes. `params` defined the custom parameters for each of the processors.
By default, processor initialization process will also init `preprocessor` attribute
which can be a processor config in itself. `preprocessor` can be then be accessed
inside the processor's functions.

Example::

    from antmmf.common.registry import registry
    from antmmf.tasks.processors import BaseProcessor


    class MyProcessor(BaseProcessor):
        def __init__(self, config, *args, **kwargs):
            return

        def __call__(self, item, *args, **kwargs):
            text = item['text']
            text = [t.strip() for t in text.split(" ")]
            return {"text": text}
"""
import inspect
import pickle
from typing import Union, Dict, Optional

import numpy as np
import torch

from antmmf.common import AntMMFConfig, Configuration
from antmmf.common.registry import registry
from antmmf.modules.utils import build_hier_tree


class BaseProcessor:
    """Every processor in antmmf needs to inherit this class for compatability
    with antmmf. End user mainly needs to implement ``__call__`` function. AntMMF
    introduces a new style processor config for clarifying each processor params'
    necessity and functionality with a dataclass inherit from AntMMFConfig. e.g.

    class DummyProcessor(BaseProcessor):
        @dataclass
        class Config(AntMMFConfig):
           dummy_param_1: List[int] = None # this is a dummy param
           dummy_param_2:int = 1333 # this is a dummy param too

        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            # The DummyProcessor params are updated with kwargs, and only params in DummyProcessor.
            # Config can be accessed. use self.config to access DummyProcessor params.
            self.config.dummy_param_2 = 0

    Args:
        config (Dict, Configuration, AntMMFConfig): Config for this processor.

    """

    def __init__(
        self, config: Union[Dict, Configuration, AntMMFConfig], *args, **kwargs
    ):
        self._init_extras(config)
        self._init_processor_config(config, **kwargs)

    def _init_processor_config(
        self, config: Union[Dict, Configuration, AntMMFConfig], **kwargs
    ):
        if hasattr(self.__class__, "Config"):
            config_class = getattr(self.__class__, "Config")
            if issubclass(config_class, AntMMFConfig):
                self.config = self.__class__.Config.create_from(config, **kwargs)

    def _init_extras(self, config, *args, **kwargs):
        self.writer = registry.get("writer")
        self.preprocessor = None

        if hasattr(config, "preprocessor"):
            self.preprocessor = Processor(config.preprocessor, *args, **kwargs)

            if self.preprocessor is None:
                raise ValueError(
                    "No processor named {} is defined.".format(config.preprocessor)
                )

    def __call__(self, item, *args, **kwargs):
        """Main function of the processor. Takes in a dict and returns back
        a dict

        Args:
            item (Dict, List): Some item that needs to be processed.

        Returns:
            Dict: Processed dict.

        """
        return item

    @classmethod
    def create_from_config(
        cls, config: Optional[Union[Dict, Configuration, AntMMFConfig]] = None, **kwargs
    ):
        if config is None:
            if not hasattr(cls, "Config"):
                raise Exception(
                    "Processor is not configured with AntMMFConfig, users need to"
                    "explicitly provide Processor Config[Dict, Configuration, AntMMFConfig]"
                )
            config = cls.Config()  # get default config of processor
        return cls(config, **kwargs)


class Processor:
    """Wrapper class used by antmmf to initialized processor based on their
    ``type`` as passed in configuration. It retrieves the processor class
    registered in registry corresponding to the ``type`` key and initializes
    with ``params`` passed in configuration. All functions and attributes of
    the processor initialized are directly available via this class.

    Args:
        config (Configuration): Configuration containing ``type`` of the processor to
                             be initialized and ``params`` of that procesor.

    """

    def __init__(self, config, *args, **kwargs):
        self.writer = registry.get("writer")

        if not hasattr(config, "type"):
            raise AttributeError(
                "Config must have 'type' attribute to specify type of processor"
            )
        processor_class = registry.get_processor_class(config.type)
        assert processor_class is not None, f"cannot find processor for ${config.type}"

        params = {}
        if not hasattr(config, "params"):
            self.writer.write(
                "Config doesn't have 'params' attribute to "
                "specify parameters of the processor "
                "of type {}. Setting to default".format(config.type)
            )
        else:
            params = config.params
        if len(inspect.signature(processor_class).parameters) == 2:
            # support processor class signature: *args, **config
            kwargs.update(**params)
            self.processor = processor_class(*args, **kwargs)
        else:
            # support processor class signature:
            # config, *args, **kwargs
            self.processor = processor_class(params, *args, **kwargs)

    def __call__(self, item, *args, **kwargs):
        return self.processor(item, *args, **kwargs)

    def __getattr__(self, name):
        if name in dir(self):
            return getattr(self, name)
        elif hasattr(self.processor, name):
            return getattr(self.processor, name)
        else:
            raise AttributeError(name)

    def __getstate__(self):
        return pickle.dumps(self.processor)

    def __setstate__(self, state):
        setattr(self, "processor", pickle.loads(state))


@registry.register_processor("copy")
class CopyProcessor(BaseProcessor):
    """
    Copy boxes from numpy array
    """

    def __init__(self, config, *args, **kwargs):
        self.max_length = config.max_length

    def __call__(self, item):
        blob = item["blob"]
        final_blob = np.zeros((self.max_length,) + blob.shape[1:], blob.dtype)
        final_blob[: len(blob)] = blob[: len(final_blob)]

        return {"blob": torch.from_numpy(final_blob)}


@registry.register_processor("hier_label_encoder")
class HierlabelProcessor(BaseProcessor):
    """
    Hierarachical Softmax label encoder, returning the path from tree root to the input label's corresponding node.
    Example::
        hier_label_processor:
            type: hier_label_encoder
            params:
              hier_label_schema:  ['教育', '科技', {'汽车':['用车技巧', '新技术汽车', '二手车']}, '体育']
    """

    def __init__(self, config, *args, **kwargs):
        self.tree = build_hier_tree(config.hier_label_schema)
        self.config = config.hier_label_schema
        self.use_multilabel = config.get("use_multilabel", False)

    def __call__(self, item):
        """
        Args:
            item: must have 'hier_label' key, whose corresponding value is a label or
                  hier-label with '-' concatenated, eg. '时尚-用车技巧' or '时尚'.

        Returns:
            hier_label: the path from root to node, aka hierarchical label for label_str.
            hier_param: the softmax param group used from root to node.

        """
        hier_label_str = item["hier_label"]
        if self.use_multilabel:
            (
                padded_hier_label,
                padded_hier_param,
                padded_hier_label_num,
            ) = self.tree.encode_multilabel_str(hier_label_str)
            return {
                "hier_label": padded_hier_label,
                "hier_param": padded_hier_param,
                "hier_label_num": padded_hier_label_num,
            }
        else:
            padded_hier_label, padded_hier_param = self.tree.encode_label_str(
                hier_label_str
            )
            return {
                "hier_label": padded_hier_label,
                "hier_param": padded_hier_param,
                "hier_label_num": None,
            }
