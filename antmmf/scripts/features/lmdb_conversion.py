# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

"""
Convert feature extracted from  antmmf.datasets.features.vision.feature_saver to lmdb format. To load features from
lmdb is much faster than directly read npy-files from disks during training.
# Refer to: https://github.com/facebookresearch/mmf/blob/master/tools/scripts/features/lmdb_conversion.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import pickle

import lmdb
import numpy as np
import tqdm

from antmmf.utils.file_io import PathManager


class LMDBConversion:
    def __init__(self):
        self.args = self.get_parser().parse_args()

    def get_parser(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument(
            "--mode",
            required=True,
            type=str,
            help="Mode can either be `convert` (for conversion of \n"
            + "features to an LMDB file) or `extract` (extract \n"
            + "raw features from a LMDB file)",
        )
        parser.add_argument(
            "--lmdb_path", required=True, type=str, help="LMDB file path"
        )
        parser.add_argument(
            "--features_folder", required=True, type=str, help="Features folder"
        )
        return parser

    def convert(self):
        env = lmdb.open(self.args.lmdb_path, map_size=1e12)  # 生成一个数据文件，定义最大空间
        id_list = []
        all_features = glob.glob(
            os.path.join(self.args.features_folder, "**", "*.npy"), recursive=True
        )

        features = []
        for feature in all_features:
            if not feature.endswith("_info.npy"):
                features.append(feature)

        # LMDB writes are very sensitive to order -
        # Sort the data before insertion speed will improve significantly
        # see detail at: https://stackoverflow.com/questions/31649216/writing-data-to-lmdb-with-python-very-slow
        features = sorted(features)

        with env.begin(write=True) as txn:
            for infile in tqdm.tqdm(features, total=len(features)):
                try:
                    reader = np.load(infile, allow_pickle=True)
                except (OSError, ValueError):
                    print("Exception:", infile)
                    continue
                item = {}
                # global unique identifier for a train/val/test sample
                split = os.path.relpath(infile, self.args.features_folder).split(
                    ".npy"
                )[0]
                item["feature_path"] = split
                key = split.encode()
                id_list.append(key)
                item["features"] = reader
                info_file = infile.split(".npy")[0] + "_info.npy"
                if not os.path.isfile(info_file):
                    txn.put(key, pickle.dumps(item))
                    continue
                # combine feature and feature info file
                try:
                    reader = np.load(info_file, allow_pickle=True)
                except (OSError, ValueError):
                    print("Exception:", info_file)
                    continue
                item["image_height"] = reader.item().get("image_height")
                item["image_width"] = reader.item().get("image_width")
                item["num_boxes"] = reader.item().get("num_boxes")
                item["objects"] = reader.item().get("objects")
                item["cls_prob"] = reader.item().get("cls_prob", None)
                item["bbox"] = reader.item().get("bbox")

                txn.put(key, pickle.dumps(item))

            txn.put(b"keys", pickle.dumps(id_list))

    def extract(self):
        os.makedirs(self.args.features_folder, exist_ok=True)
        env = lmdb.open(
            self.args.lmdb_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            _image_ids = pickle.loads(txn.get(b"keys"))
            for img_id in tqdm.tqdm(_image_ids):
                item = pickle.loads(txn.get(img_id))
                img_id = img_id.decode("utf-8")
                tmp_dict = {}
                tmp_dict["image_id"] = img_id
                tmp_dict["bbox"] = item["bbox"]
                tmp_dict["num_boxes"] = item["num_boxes"]
                tmp_dict["image_height"] = item["image_height"]
                tmp_dict["image_width"] = item["image_width"]
                tmp_dict["objects"] = item["objects"]
                tmp_dict["cls_prob"] = item["cls_prob"]

                info_file_base_name = str(img_id) + "_info.npy"
                file_base_name = str(img_id) + ".npy"

                path = os.path.join(self.args.features_folder, file_base_name)
                if PathManager.exists(path):
                    continue
                info_path = os.path.join(self.args.features_folder, info_file_base_name)
                base_path = "/".join(path.split("/")[:-1])
                PathManager.mkdirs(base_path)
                np.save(PathManager.open(path, "wb"), item["features"])
                np.save(PathManager.open(info_path, "wb"), tmp_dict)

    def execute(self):
        if self.args.mode == "convert":
            self.convert()
        elif self.args.mode == "extract":
            self.extract()
        else:
            raise ValueError("mode must be either `convert` or `extract` ")


if __name__ == "__main__":
    lmdb_converter = LMDBConversion()
    lmdb_converter.execute()
