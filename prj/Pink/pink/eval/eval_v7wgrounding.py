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
    correct = 0
    sum = 0
    for pred in predictions:
        gt_ans = pred['gt_ans']
        if gt_ans == pred['answer']:
            correct += 1
        sum += 1
    print("num samples: {}".format(sum))
    print("accuracy: {:.4f}".format(correct / sum))
