from pycocotools.coco import COCO
import json

annFile='/gruntdata/nas018/workspace/licong/data/coco/annotations/instances_train2017.json'
coco=COCO(annFile)

img_ids = coco.getImgIds()
data_list = []
for img_id in img_ids:
    result_dict = {}
    raw_img_info = coco.getImgIds([img_id])
    img = coco.loadImgs(raw_img_info)[0]
    result_dict['id'] = img['id']
    result_dict['image_id'] = img['file_name']
    result_dict['height'] = img['height']
    result_dict['width'] = img['width']

    ann_ids = coco.getAnnIds(imgIds=[img_id])
    raw_ann_info = coco.loadAnns(ann_ids)
    result_dict['anno'] = []
    for ann in raw_ann_info:
        ann['category_name'] = coco.loadCats([ann['category_id']])[0]['name']
        x1, y1, w, h = ann['bbox']
        inter_w = max(0, min(x1 + w, result_dict['width']) - max(x1, 0))
        inter_h = max(0, min(y1 + h, result_dict['height']) - max(y1, 0))
        if inter_w * inter_h == 0:
            continue
        if ann['area'] <= 0 or w < 1 or h < 1:
            continue
        bbox = [x1, y1, min(x1 + w, result_dict['width']), min(y1 + h, result_dict['height'])]
        ann['bbox'] = bbox
        if ann.get('iscrowd', False):
            ann['ignore'] = 1
        else:
            ann['ignore'] = 0
        # NOTE: bounding box is w, h, width, height now
        result_dict['anno'].append(ann)
    data_list.append(result_dict)

json.dump(data_list, open("/mnt/workspace/xuanshiyu.xsy/instruct/instruct_coco_train2017.json", "w"))