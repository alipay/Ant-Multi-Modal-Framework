from typing import Union

import torch
from nn4k.executor import LLMExecutor


class M2EncoderExecutor(LLMExecutor):

    @classmethod
    def from_config(cls, nn_config: Union[str, dict]) -> "M2EncoderExecutor":
        executor = cls(nn_config)
        return executor

    def load_model(self, args=None, mode=None, **kwargs):
        from nn4k.consts import NN_DEVICE_KEY
        from nn4k.utils.config_parsing import get_string_field

        nn_config: dict = args or self.init_args
        if self._model is None:
            model_config = get_string_field(nn_config, 'model_config', '')
            device = nn_config.get(NN_DEVICE_KEY)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device

            from vlmo.utils.beit_utils import load_from_config
            model, processors = load_from_config(model_config)
            model.to(device).eval()
            self._model = model
            self._tokenizer, self._img_processor = processors

    def inference(self, data, args=None, **kwargs):
        from PIL import Image
        img = self._img_processor(Image.open(data['image_path'])).unsqueeze(0)
        img_data = {"image": [img.to(self._device)]}
        txt_encoding = self._tokenizer(
            data['texts'],
            padding="max_length",
            truncation=True,
            max_length=self._model.hparams.config["max_text_len"],
            return_special_tokens_mask=True,
        )
        txt_data = {
            "text_ids": torch.tensor(txt_encoding["input_ids"]).to(self._device),
            "text_masks": torch.tensor(txt_encoding["attention_mask"]).to(self._device),
            "text_labels": None,
        }
        # 提取图像/文本特征或计算图文相似性
        with torch.no_grad():
            img_feats = self._model.infer_image(img_data)["cls_vlffn_feats"]
            txt_feats = self._model.infer_text(txt_data)["cls_vlffn_feats"]

            # 计算图文相似性
            logit_scale = self._model.logit_scale.exp()
            logits_per_img = logit_scale * img_feats @ txt_feats.t()

            probs = logits_per_img.softmax(dim=-1).cpu().numpy()
            # print("Label probs:", probs)  # [[3.64e-05 1.80-04 1.87e-04 9.99e-01]], [[0 0 0 1]]
            return probs
