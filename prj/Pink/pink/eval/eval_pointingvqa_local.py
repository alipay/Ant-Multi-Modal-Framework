import argparse
import json
import os
import re
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--crop-size', type=int, default=224)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    predictions = [json.loads(line) for line in open(args.result_file)]
    point_correct = 0
    bbox_correct = 0
    sum = 0
    for pred in predictions:
        gt_ans = pred['gt_ans']
        if gt_ans == pred['answer_bbox'].lower():
            bbox_correct += 1
        if gt_ans == pred['answer_point'].lower():
            point_correct += 1
        sum += 1
    print("num samples: {}".format(sum))
    print("bbox accuracy: {:.4f}".format(bbox_correct / sum))
    print("point accuracy: {:.4f}".format(point_correct / sum))
