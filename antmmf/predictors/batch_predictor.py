# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent.futures import ThreadPoolExecutor

from antmmf.common.report import default_result_formater
from antmmf.predictors.base_predictor import BasePredictor
from antmmf.structures import Sample, SampleList


class BatchPredictor(BasePredictor):
    """
    predictor支持batch_inference需要做以下修改:
    predictor侧修改:
        1) 修改predictor基类：BasePredictor -> BatchPredictor
        2) dummy_request: 由单样本输入修改为返回batch输入，
           format_result: 对整个batch结果进行格式化输出
    可参考: MMBTBatchPredictor

    service侧batch请求数据格式为:
    {"urlList": [url1, url2, ..,],
     "content": {"result": [{"text": text1}, {"text": text2}, ...]} }
    """

    DEFAULT_SAMPLE_WORKERS = 3

    def predict(self, data=None, json_obj=None):
        """
        Args:
            data(list): list of image bytes
            json_obj(dict): with 'result' key, whose corresponding value
                               is a list of dict containing multi-modal input info.
                               eg.{"result": [{"text": "狗"}, {"text": "和朋友的快乐时光"}]

        Returns:
                Report(OrderedDict)
        """
        if data is None:
            data, json_obj = self.dummy_request()

        self.profiler.reset()
        if "result" not in json_obj:
            json_obj["result"] = [{} for _ in range(len(data))]
        if len(json_obj["result"]) > 0:
            assert len(data) == len(
                json_obj["result"]
            ), "number of input images should equal to length of json_obj['result']"

        num_workers = self.predictor_parameters.get(
            "max_workers", BatchPredictor.DEFAULT_SAMPLE_WORKERS
        )
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            samples = list(
                executor.map(
                    lambda x: self._build_sample(*x),
                    zip(data, json_obj.get("result", [])),
                )
            )

        self.profile("Build sample time")
        if not isinstance(samples[0], Sample):
            raise Exception(
                f"Method _build_sample is expected to return a instance of antmmf.structures.sample.Sample,"
                f"but got type {type(samples[0])} instead."
            )

        # batch inference
        result = self._predict(SampleList(samples))
        np_result = default_result_formater(result)
        result = self.format_result(np_result)
        assert isinstance(
            result, dict
        ), f"Result should be instance of Dict,but got f{type(result)} instead"
        return result
