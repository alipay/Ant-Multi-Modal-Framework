import json

dataset_dict = json.load(open("/Users/xuan/Downloads/dataset_v7w_telling.json", "r"))

test_split = []
val_split = []
train_split = []

for image in dataset_dict['images']:
    image_id = image["image_id"]
    split = image['split']
    file_name = image['filename']
    merge_array = []
    for qa_pair in image['qa_pairs']:
        qa_pair["image_id"] = image_id
        qa_pair["file_name"] = file_name
        merge_array.append(qa_pair)
    if split == "train":
        train_split += merge_array
    elif split == "val":
        val_split += merge_array
    elif split == "test":
        test_split += merge_array
print(len(train_split))
print(len(test_split))
print(len(val_split))

json.dump(train_split, open("/Users/xuan/Downloads/visual_7w_telling/train_split.json", "w"))
json.dump(test_split, open("/Users/xuan/Downloads/visual_7w_telling/test_split.json", "w"))
json.dump(val_split, open("/Users/xuan/Downloads/visual_7w_telling/val_split.json", "w"))