import json

dataset_dict = json.load(open("/Users/xuan/Downloads/dataset_v7w_pointing.json", "r"))

test_split = []
val_split = []
train_split = []

bbox_dict = {}
for bbox in dataset_dict['boxes']:
    bbox_dict[bbox['box_id']] = bbox

for image in dataset_dict['images']:
    image_id = image["image_id"]
    split = image['split']
    file_name = image['filename']
    merge_array = []
    for qa_pair in image['qa_pairs']:
        qa_pair['answer_box'] = bbox_dict[qa_pair['answer']]
        qa_pair['multiple_choices_box'] = []
        for box in qa_pair['multiple_choices']:
            qa_pair['multiple_choices_box'].append(bbox_dict[box])
        qa_pair["image_id"] = image_id
        qa_pair["file_name"] = file_name
        merge_array.append(qa_pair)
    if split == "train":
        train_split += merge_array
    elif split == "val":
        val_split += merge_array
    elif split == "test":
        test_split += merge_array

json.dump(train_split, open("/Users/xuan/Downloads/visual_7w/train_split.json", "w"))
json.dump(test_split, open("/Users/xuan/Downloads/visual_7w/test_split.json", "w"))
json.dump(val_split, open("/Users/xuan/Downloads/visual_7w/val_split.json", "w"))