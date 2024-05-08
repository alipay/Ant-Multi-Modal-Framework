# Copyright (c) Alibaba, Inc. and its affiliates.

import torch

from modelscope.models.base import TorchModel
from modelscope.preprocessors.base import Preprocessor
from modelscope.pipelines.base import Model, Pipeline
from modelscope.utils.config import Config
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.models.builder import MODELS


from nn4k.consts import NN_EXECUTOR_KEY
from nn4k.invoker import LLMInvoker
from vlmo.utils.beit_utils import load_from_config


@MODELS.register_module("multi-modal-embedding-task", module_name="m2-encoder")
class MyCustomModel(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        # self.model = self.init_model(**kwargs)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model_config = "./configs/Encoder_0.4B.json"
        model, processors = load_from_config(model_config)
        self.model = model
        self.model.to(self._device).eval()
        self._tokenizer, self._img_processor = processors

    def forward(self, forward_params):
        rets = {}
        if "text" in forward_params:
            text = forward_params.get("text")
            txt_encoding = self._tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.model.hparams.config["max_text_len"],
                return_special_tokens_mask=True,
            )
            txt_data = {
                "text_ids": torch.tensor(txt_encoding["input_ids"]).to(self._device),
                "text_masks": torch.tensor(txt_encoding["attention_mask"]).to(
                    self._device
                ),
                "text_labels": None,
            }
            txt_feats = self.model.infer_text(txt_data)["cls_vlffn_feats"]
            rets.update({"text_embedding": txt_feats.detach()})
        if "img" in forward_params:
            input_img = forward_params["img"]
            img = self._img_processor(input_img).unsqueeze(0)
            img_data = {"image": [img.to(self._device)]}
            img_feats = self.model.infer_image(img_data)["cls_vlffn_feats"]
            rets.update({"img_embedding": img_feats.detach()})

        return rets


@PREPROCESSORS.register_module(
    "multi-modal-embedding-task", module_name="m2-encoder-preprocessor"
)
class MyCustomPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainsforms = self.init_preprocessor(**kwargs)

    def __call__(self, results):
        return self.trainsforms(results)

    def init_preprocessor(self, **kwarg):
        """Provide default implementation based on preprocess_cfg and user can reimplement it.
        if nothing to do, then return lambda x: x
        """
        return lambda x: x


@PIPELINES.register_module(
    "multi-modal-embedding-task", module_name="multi-modal-embedding-pipeline"
)
class MyCustomPipeline(Pipeline):
    """Give simple introduction to this pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline
    >>> input = "Hello, ModelScope!"
    >>> my_pipeline = pipeline('my-task', 'my-model-id')
    >>> result = my_pipeline(input)

    """

    def __init__(self, model, preprocessor=None, **kwargs):
        """
        use `model` and `preprocessor` to create a custom pipeline for prediction
        Args:
            model: model id on modelscope hub.
            preprocessor: the class of method be init_preprocessor
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

        assert isinstance(model, str) or isinstance(
            model, Model
        ), "model must be a single str or Model"
        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError
        pipe_model.eval()

        if preprocessor is None:
            preprocessor = MyCustomPreprocessor()
        super().__init__(model=pipe_model, preprocessor=preprocessor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method
        considered to be a normal classmethod with default implementation / output

        Default Returns:
            Dict[str, str]:  preprocess_params = {}
            Dict[str, str]:  forward_params = {}
            Dict[str, str]:  postprocess_params = pipeline_parameters
        """
        return {}, pipeline_parameters, {}

    def _check_input(self, inputs):
        pass

    def _check_output(self, outputs):
        pass

    def forward(self, forward_params):
        """Provide default implementation using self.model and user can reimplement it"""
        return super().forward(forward_params)

    def postprocess(self, inputs):
        """If current pipeline support model reuse, common postprocess
            code should be write here.

        Args:
            inputs:  input data

        Return:
            dict of results:  a dict containing outputs of model, each
                output should have the standard output name.
        """
        return inputs


# Tips: usr_config_path is the temporary save configuration location， after upload modelscope hub, it is the model_id
usr_config_path = "/tmp/snapdown/"
config = Config(
    {
        "framework": "pytorch",
        "task": "multi-modal-embedding-task",
        "model": {"type": "m2-encoder"},
        "pipeline": {"type": "multi-modal-embedding-pipeline"},
        "allow_remote": True,
    }
)
config.dump("/tmp/snapdown/" + "configuration.json")

if __name__ == "__main__":
    from modelscope.pipelines import pipeline
    from modelscope.preprocessors.image import load_image

    model = "M2Cognition/M2-Encoder"
    pipe = pipeline("multi-modal-embedding-task", model=model)
    input_texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]
    inputs = {"text": input_texts}
    output = pipe(inputs)["text_embedding"]
    print("output", output)
    input_img = load_image(
        "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
    )  # 支持皮卡丘示例图片路径/本地图片 返回PIL.Image
    inputs = {"img": input_img}
    img_embedding = pipe(inputs)["img_embedding"]  # 2D Tensor, [图片数, 特征维度]
    print("output", output)
