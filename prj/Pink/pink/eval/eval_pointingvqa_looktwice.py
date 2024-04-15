import argparse
import json
import os
import re
import random
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--crop-size', type=int, default=224)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    predictions = [json.loads(line) for line in open(args.result_file)]
    point_correct = {'obj_question': 0, 'super_question': 0, 'general_question': 0}
    bbox_correct = {'obj_question': 0, 'super_question': 0, 'general_question': 0}
    most_correct = 0

    option_point_correct = {'obj_question': 0, 'super_question': 0, 'general_question': 0}
    option_bbox_correct = {'obj_question': 0, 'super_question': 0, 'general_question': 0}
    option_most_correct = 0
    sum = 0
    question_type = ['obj_question', 'super_question', 'general_question']
    for pred in predictions:
        gt_ans = pred['gt_ans']
        answers = []
        for select_type in question_type:
            answers.append(pred['{}_answer_bbox'.format(select_type)])
            answers.append(pred['{}_answer_point'.format(select_type)])
            if gt_ans == pred['{}_answer_bbox'.format(select_type)]:
                bbox_correct[select_type] += 1
                option_bbox_correct[select_type] += 1
            else:
                try:
                    if int(gt_ans) >= 2:
                        if int(pred['{}_answer_bbox'.format(select_type)]) >= 2:
                            option_bbox_correct[select_type] += 1
                except:
                    pass

            if gt_ans == pred['{}_answer_point'.format(select_type)]:
                point_correct[select_type] += 1
                option_point_correct[select_type] += 1
            else:
                try:
                    if int(gt_ans) >= 2:
                        if int(pred['{}_answer_point'.format(select_type)]) >= 2:
                            option_point_correct[select_type] += 1
                except:
                    pass

        counter = Counter(answers)
        most_answer = counter.most_common(1)[0][0]
        if gt_ans == most_answer:
            most_correct += 1
            option_most_correct += 1
        # elif int(gt_ans) >= 2 and int(most_answer) >= 2:
        #     option_most_correct += 1
        sum += 1
    print("num samples: {}".format(sum))
    print("most select accuracy: {:.4f}".format(most_correct / sum))
    print("option most select accuracy: {:.4f}".format(option_most_correct / sum))
    for select_type in question_type:
        print("{} bbox accuracy: {:.4f}".format(select_type, bbox_correct[select_type] / sum))
        print("option {} bbox accuracy: {:.4f}".format(select_type, option_bbox_correct[select_type] / sum))
        print("{} point accuracy: {:.4f}".format(select_type, point_correct[select_type] / sum))
        print("option {} point accuracy: {:.4f}".format(select_type, option_point_correct[select_type] / sum))
