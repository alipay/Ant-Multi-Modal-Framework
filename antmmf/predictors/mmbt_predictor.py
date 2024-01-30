# Copyright (c) 2023 Ant Group and its affiliates.

import io

import numpy as np
from PIL import Image

from antmmf.common.constants import TEXT_MODALITY, IMAGES_STR
from antmmf.common.registry import registry
from antmmf.datasets.mm_dataset import MmfImageTextDataset
from antmmf.predictors.base_predictor import BasePredictor
from antmmf.predictors.batch_predictor import BatchPredictor
from antmmf.utils.general import get_absolute_path


@registry.register_predictor("MMBTPredictor")
class MMBTPredictor(BasePredictor):
    def dummy_request(self):
        pil_image_path = get_absolute_path("../tests/data/image/dog.jpg")
        bytes_image = open(pil_image_path, "rb").read()
        json_input = {"img_name": "dog.jpg", "result": {"text": "it is a dog"}}
        return bytes_image, json_input

    def _build_sample(self, data, json_obj):
        # bytes string for online request
        image_str = data

        # restore pil_image from image_str
        pil_image = Image.open(io.BytesIO(bytes(image_str)))
        if not pil_image.mode == "RGB":
            pil_image = pil_image.convert("RGB")

        text_sample_info = {TEXT_MODALITY: json_obj["result"][TEXT_MODALITY]}
        image_sample_info = {IMAGES_STR: [pil_image]}

        current_sample = MmfImageTextDataset.build_sample_with_text_image(
            text_sample_info,
            self.processors.text_processor,
            image_sample_info,
            self.processors.test_image_processor,
        )

        return current_sample

    def format_result(self, report):
        probs = report["prob"][0]
        label = probs.argmax(-1)
        prob = np.around(probs[label], decimals=4)
        return {"prob": prob, "label": label, "logits": report["logits"]}


@registry.register_predictor("MMBTBatchPredictor")
class MMBTBatchPredictor(BatchPredictor):
    def dummy_request(self):
        pil_image_path = [
            "../tests/data/predictors/samples/images/dog.jpg",
            "../tests/data/predictors/samples/images/dog.jpg",
        ]
        bytes_images = []
        for image_file in pil_image_path:
            bytes_images.append(open(get_absolute_path(image_file), "rb").read())
        json_input = {
            "result": [
                {"text": "狗"},
                {"text": "和朋友的快乐时光"},
            ]
        }
        return bytes_images, json_input

    def _build_sample(self, data, json_obj):
        # restore pil_image
        pil_image = Image.open(io.BytesIO(bytes(data)))
        if not pil_image.mode == "RGB":
            pil_image = pil_image.convert("RGB")

        text_sample_info = {TEXT_MODALITY: json_obj[TEXT_MODALITY]}
        image_sample_info = {IMAGES_STR: [pil_image]}

        current_sample = MmfImageTextDataset.build_sample_with_text_image(
            text_sample_info,
            self.processors.text_processor,
            image_sample_info,
            self.processors.test_image_processor,
        )

        return current_sample

    def format_result(self, report):
        probs = report["prob"]
        labels = probs.argmax(-1)
        prob = np.around(probs[np.arange(probs.shape[0]), labels], decimals=4)
        return {"probs": prob.tolist(), "label": labels.tolist()}


if __name__ == "__main__":
    from antmmf.predictors.build import build_online_predictor

    model_dir = "../prj/attrim/save/attrim_mmbt"
    predictor = build_online_predictor(model_dir)

    predictor.load(with_ckpt=True)
    # predictor.export_onnx()
    result1 = predictor.predict()
    result2 = predictor.predict_onnx()
    print("pytorch inference:", result1)
    print("onnx inference:", result2)
