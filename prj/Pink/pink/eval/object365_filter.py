import argparse
import json
import os
import re
import random
from tqdm import tqdm


def computeIoU(box1, box2):
    # each box is of [x1, y1, x2, y2]
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-file', type=str)
    return parser.parse_args()


# {"id": 900001, "image_id": "patch16/objects365_v2_00900001.jpg", "anno": [{"id": 97, "iscrowd": 0, "isfake": 0, "area": 49124.52177485057, "isreflected": 0, "bbox": [0, 187.4827881, 157.0656127977, 500], "image_id": 900001, "category_id": 1, "category_name": "Person", "ignore": 0}, {"id": 98, "iscrowd": 0, "isfake": 0, "area": 1860.6347378153857, "isreflected": 0, "bbox": [49.3554382236, 296.94555665, 113.7900085542, 325.8218994], "image_id": 900001, "category_id": 33, "category_name": "Necklace", "ignore": 0}, {"id": 99, "iscrowd": 0, "isfake": 0, "area": 21967.07117695177, "isreflected": 1, "bbox": [186.0563354553, 191.00616455, 300.44750977530003, 383.04083249999996], "image_id": 900001, "category_id": 1, "category_name": "Person", "ignore": 0}, {"id": 100, "iscrowd": 0, "isfake": 0, "area": 468.88838629800705, "isreflected": 1, "bbox": [212.7305297934, 262.5275879, 251.2655639523, 274.69543455], "image_id": 900001, "category_id": 33, "category_name": "Necklace", "ignore": 0}, {"id": 101, "iscrowd": 0, "isfake": 0, "area": 73224.4753078346, "isreflected": 0, "bbox": [129.100952151, 59.729492199999996, 333, 418.26922605], "image_id": 900001, "category_id": 80, "category_name": "Mirror", "ignore": 0}], "pred": [{"object_id": 0, "caption": "girl with long hair", "bbox": [177.489, 198.0, 309.024, 381.0], "format_error": 0}, {"object_id": 2, "caption": "a woman in a red dress", "bbox": [0.999, 188.0, 154.845, 494.0], "format_error": 0}, {"object_id": 4, "caption": "a mirror on the wall", "bbox": [136.863, 73.0, 332.001, 405.0], "format_error": 0}]}

if __name__ == "__main__":
    args = get_args()
    iou_threshold = 0.5

    loc_pattern = re.compile(r"(\[[0-9].[0-9][0-9][0-9],[0-9].[0-9][0-9][0-9],[0-9].[0-9][0-9][0-9],[0-9].[0-9][0-9][0-9]\])")
    bbox_pattern = re.compile(r"[0-9].[0-9][0-9][0-9]")
    warning_pattern = re.compile(r"(\[[0-9].*[0-9]{0,3}(?:,+[0-9].*[0-9]{0,3}){0,3})")

    output_file = open(args.output_file, "w")
    before_filter_num = 0
    after_filter_num = 0
    object_before_filter_num = 0
    object_after_filter_num = 0
    total_caption_length = 0

    with open(args.result_file, "r") as f:
        for line in tqdm(f):
            before_filter_num += 1
            result_dict = json.loads(line)
            if len(result_dict['pred']) == 0:
                continue
            correct_pred = []
            for pred in result_dict['pred']:
                object_before_filter_num += 1
                if pred['format_error'] == 1:
                    continue
                object_id = pred['object_id']
                gt_object = result_dict['anno'][object_id]
                iou = computeIoU(pred['bbox'], gt_object['bbox'])
                if iou >= iou_threshold:
                    pred['bbox'] = gt_object['bbox']
                    correct_pred.append(pred)
                    object_after_filter_num += 1
                    total_caption_length += len(pred['caption'].split(" "))
            for anno in result_dict['anno']:
                anno['category_name'] = anno['category_name'].lower()

            grounding_caption = result_dict['grounding_caption']
            bbox_list = loc_pattern.findall(grounding_caption)

            split_caption = grounding_caption.split(" ")
            
            shift_index = 0
            part_caption = []
            part_object = []
            start_caption = True
            part_index = -1
            for word in split_caption:
                if shift_index != len(bbox_list) and word == bbox_list[shift_index]:
                    shift_index += 1
                    part_object[part_index].append(word)
                    start_caption = True
                else:
                    if start_caption:
                        part_index += 1
                        part_caption.append([])
                        part_object.append([])
                    part_caption[part_index].append(word)
                    start_caption = False
            remap_objects = []
            for objects in part_object:
                org_objects = {}
                for object in objects:
                    bbox_token = bbox_pattern.findall(object)
                    if len(bbox_token) == 4:
                        bbox = [float(bbox_token[0]) * result_dict['orig_width'], float(bbox_token[1]) * result_dict['orig_height'], float(bbox_token[2]) * result_dict['orig_width'], float(bbox_token[3]) * result_dict['orig_height']]
                    max_iou = 0.0
                    max_iou_object = None
                    for obj in result_dict['anno']:
                        if obj['ignore'] == 1:
                            continue
                        iou = computeIoU(obj['bbox'], bbox)
                        if iou > max_iou:
                            max_iou = iou
                            max_iou_object = obj
                    if max_iou_object is not None and max_iou > iou_threshold:
                        if max_iou_object['id'] not in org_objects.keys():
                            org_objects[max_iou_object['id']] = (max_iou_object, max_iou)
                        else:
                            if org_objects[max_iou_object['id']][1] < max_iou:
                                org_objects[max_iou_object['id']] = (max_iou_object, max_iou)
                remap_objects.append([o[0] for o in org_objects.values()])
            empty_object = 0
            for e in remap_objects:
                if len(e) == 0:
                    empty_object += 1
            empty_raio = empty_object / len(remap_objects)
            
            grounding_caption_text = ""
            generate_caption = ""
            for part_index, part_text in enumerate(part_caption):
                if len(remap_objects[part_index]) > 0:
                    grounding_caption_text += " ".join(part_text) + " <ph_ed> "
                else:
                    grounding_caption_text += " ".join(part_text) + " "
                generate_caption += " ".join(part_text) + " "
            grounding_caption_text = grounding_caption_text.rstrip(" <ph_ed> ")
            generate_caption = generate_caption.rstrip(" ")

            extra = warning_pattern.findall(generate_caption)
            if len(extra) > 0:
                for e in extra:
                    generate_caption = generate_caption.replace(e, "")
                    grounding_caption_text = grounding_caption_text.replace(e, "")
                generate_caption = generate_caption + "."
                grounding_caption_text = grounding_caption_text + "."
            result_dict['generate_caption'] = generate_caption
            result_dict['generate_gounding_caption'] = grounding_caption_text
            result_dict['maped_object'] = remap_objects
            result_dict['pred'] = correct_pred

            if len(correct_pred) != 0:
                after_filter_num += 1
            else:
                continue
            result_dict['pred'] = correct_pred
            output_file.write(json.dumps(result_dict) + "\n")
            
    output_file.close()
    print("before filter num: {}".format(before_filter_num))
    print("after filter num: {}".format(after_filter_num))
    print("object before filter num: {}".format(object_before_filter_num))
    print("object after filter num: {}".format(object_after_filter_num))
    print("avg description length: {}".format(total_caption_length / object_after_filter_num))
