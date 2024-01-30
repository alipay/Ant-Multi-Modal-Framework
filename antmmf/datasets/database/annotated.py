# -*- coding: utf-8 -*-
# Copyright (c) 2023 Ant Group and its affiliates.

import glob
import json
import os.path as osp
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from antmmf.common import AntMMFConfig


class AnnotatedDatabase(torch.utils.data.Dataset):
    """
    Dataset for IMDB used in antmmf
    General format that we have standardize follows:
    {
        metadata: {
            'version': x
        },
        data: [
            {
                'id': DATASET_SET_ID,
                'set_folder': <directory>,
                'feature_path': <file_path>,
                'info': {
                    // Extra information
                    'questions_tokens': [],
                    'answer_tokens': []
                }
            }
        ]
    }
    """

    def __init__(self, annotation_path, anno_type="normal", **kwargs):
        """
        :param annotation_path:
        :param anno_type: handle file with same suffix but using different analyzing method.
        """
        super().__init__()
        self.kwargs = kwargs
        self._load_annotation_db(annotation_path, anno_type)
        self.data = self.process_annotation(self.data)

    def _load_annotation_db(self, imdb_path, anno_type):
        if isinstance(imdb_path, list):
            assert len(imdb_path) == 1, "only support reading annotation one by one"
            self._load_annotation_db(imdb_path[0], anno_type)
        elif osp.isdir(
            imdb_path
        ):  # imdb_path is a dir containing multiple jsonl-format files
            jsonl_files = glob.glob(osp.join(imdb_path, "*.jsonl"))
            self._load_jsonl_dir(jsonl_files)
        else:
            if imdb_path.endswith(".npy"):
                self._load_npy(imdb_path)
            elif imdb_path.endswith(".jsonl"):
                self._load_jsonl(imdb_path)
            elif imdb_path.endswith(".json"):
                self._load_json(imdb_path, anno_type)
            elif imdb_path.endswith(".csv"):
                self._load_csv(imdb_path)
            elif imdb_path.endswith(".tsv"):
                self._load_tsv(imdb_path)
            else:
                raise ValueError("Unknown file format for imdb:%s" % imdb_path)

    def _load_jsonl_dir(self, jsonl_list):
        # To load jsonl dir for aidesk format: tests/data/aidesk
        self.data = []
        for jsonl_f in jsonl_list:
            with open(jsonl_f, "r") as f:
                for js_line in f:
                    js_line = js_line.strip()
                    if not js_line:
                        continue
                    self.data.append(json.loads(js_line))
        self.metadata = {}
        self.start_idx = 0

    def _load_jsonl(self, imdb_path):
        with open(imdb_path, "r", encoding="utf-8") as f:
            db = f.readlines()
            for idx, line in enumerate(db):
                db[idx] = json.loads(line.strip("\n"))
            self.metadata = {}
            self.data = db
            self.start_idx = 0

    def _load_coco_json(self, imdb_path):
        from pycocotools.coco import COCO

        self.coco = COCO(imdb_path)
        self.data = list(sorted(self.coco.imgs.keys()))
        self.metadata = {}
        self.start_idx = 0

    def _load_normal_json(self, imdb_path):
        with open(imdb_path, "r") as f:
            db = json.load(fp=f)
            self.metadata = {}
            self.data = db
            self.start_idx = 0

    def _load_json(self, imdb_path, anno_type):
        if anno_type == "coco":
            self._load_coco_json(imdb_path)
        else:
            self._load_normal_json(imdb_path)

    def _load_csv(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.metadata = self.data.keys()
        self.start_idx = 0

    def _load_tsv(self, csv_path):
        self.data = []
        with open(csv_path, encoding="utf-8") as fd:
            idx = 0
            for ln in fd:
                ln = ln.strip()
                if idx == 0:
                    tag = ln.split("\t")
                else:
                    self.data.append(
                        dict([(t, o.strip()) for t, o in zip(tag, ln.split("\t"))])
                    )
                idx += 1
        self.metadata = tag
        self.start_idx = 0

    def _load_npy(self, imdb_path):
        self.db = np.load(imdb_path, allow_pickle=True)
        self.start_idx = 0

        if type(self.db) == dict:
            self.metadata = self.db.get("metadata", {})
            self.data = self.db.get("data", [])
        else:
            # TODO: Deprecate support for this
            self.metadata = {"version": 1}
            self.data = self.db
            # Handle old imdb support
            if "image_id" not in self.data[0]:
                self.start_idx = 1

        if len(self.data) == 0:
            self.data = self.db

    def __len__(self):
        return len(self.data) - self.start_idx

    def process_annotation(self, annotation_database):
        return annotation_database

    def preprocess_item(self, item):
        # Backward support to older IMDBs
        if "answers" not in item:
            if "all_answers" in item and "valid_answers" not in item:
                item["answers"] = item["all_answers"]
            if "valid_answers" in item:
                item["answers"] = item["valid_answers"]

        # TODO: Later clean up VizWIz IMDB from copy tokens
        if "answers" in item and item["answers"][-1] == "<copy>":
            item["answers"] = item["answers"][:-1]

        if "answers" in item:
            item["label"] = item["answers"]
        return item

    def __getitem__(self, idx):
        data = self.data[idx + self.start_idx]
        return self.preprocess_item(data)

    def get_version(self):
        return self.metadata.get("version", None)


class DetectionCOCOAnnotated(AnnotatedDatabase):
    """
    Annotated Database handles loading coco annotation
    """

    @dataclass
    class Config(AntMMFConfig):
        annotation_path: str = None

    def __init__(self, config, *args, **kwargs):
        self.config = DetectionCOCOAnnotated.Config.create_from(config, **kwargs)
        super().__init__(self.config.annotation_path, anno_type="coco")
        self.id2imginfo = self.coco.imgs
        self.img2id = dict(
            [
                (img_info["file_name"], img_id)
                for img_id, img_info in self.id2imginfo.items()
            ]
        )

    def get_annotation_by_name(self, img_name):
        img_id = self.img2id.get(img_name)
        annotation = []
        if img_id is not None:  # have detection annotations
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            annotation = self.coco.loadAnns(ann_ids)
        return annotation

    def get_annotation_by_idx(self, idx):
        image_id = self.data[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotation = self.coco.loadAnns(ann_ids)
        return annotation

    def __getitem__(self, id_or_name):
        if isinstance(id_or_name, int):
            annotation = self.get_annotation_by_idx(id_or_name)
        else:
            assert isinstance(id_or_name, str)
            annotation = self.get_annotation_by_name(id_or_name)
        return annotation
