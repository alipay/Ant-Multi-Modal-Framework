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


def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2] - box1[0])*(box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return float(inter)/union


if __name__ == "__main__":
    args = get_args()

    LOCATION_TOKENS = ["<{:0>4d}>".format(i) for i in range(args.crop_size)]
    predictions = [json.loads(line) for line in open(args.result_file)]
    ans_file = open(args.result_file.split(".json_result")[0] + "_final.json", "w")
    # pattern = re.compile(r'<[0-9][0-9][0-9][0-9]>')
    pattern = re.compile(r'[0-9]\.[0-9][0-9][0-9]')
    correct = 0
    location_error = 0
    format_error = 0
    error_avg_iou = 0
    results = {'correct': [], 'incorrect': []}
    for pred in predictions:
        result = {}
        result['exp_id'] = pred['exp_id']
        result['ref_id'] = pred['ref_id']
        [orig_width, orig_height] = pred['orig_wh']
        [current_width, current_height] = pred['cur_wh']
        gt_bbox = pred['bbox']
        pred = pred['pred_bbox']
        res = pattern.findall(pred)
        pred_bbox = []
        if len(res) == 4:
            for r in res:
                # pred_bbox.append(LOCATION_TOKENS.index(r) / (len(LOCATION_TOKENS) - 1))
                pred_bbox.append(float(r))
            pred_bbox = [pred_bbox[0] * orig_width, pred_bbox[1] * orig_height, pred_bbox[2] * orig_width, pred_bbox[3] * orig_height]
            result['iou'] = computeIoU(pred_bbox, gt_bbox)
            if result['iou'] > 0.5:
                correct += 1
                results['correct'].append(result)
            else:
                error_avg_iou += result['iou']
                results['incorrect'].append(result)
                location_error += 1
        else:
            print(pred)
            result['iou'] = 0
            results['incorrect'].append(result)
            format_error += 1
    results['acc_count'] = correct
    results['format_error'] = format_error
    results['location_error'] = location_error
    assert (correct + format_error + location_error) == len(predictions)
    results['acc1'] = correct / len(predictions)
    print(results['acc1'])
    print(results['location_error'])
    print(results['format_error'])
    print(error_avg_iou / results['location_error'])
    json.dump(results, ans_file)
    ans_file.close()
    # results = {'correct': [], 'incorrect': []}
    # sqa_results = {}
    # sqa_results['acc'] = None
    # sqa_results['correct'] = None
    # sqa_results['count'] = None
    # sqa_results['results'] = {}
    # sqa_results['outputs'] = {}
