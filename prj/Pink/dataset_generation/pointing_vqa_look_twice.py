import json

split_id = json.load(open("/Users/xuan/Downloads/pointingqa-main/Datasets/LookTwiceQA/looktwiceqa_test_imgs.json", "r"))
print(len(split_id))
dataset_dict = json.load(open("/Users/xuan/Downloads/pointingqa-main/Datasets/LookTwiceQA/looktwiceqa_dataset.json", "r"))

merge_dict = []
for img_id in split_id:
    merge_dict += dataset_dict[img_id]
print(len(merge_dict))

json.dump(merge_dict, open("/Users/xuan/Downloads/pointingqa-main/Datasets/LookTwiceQA/test_split.json", "w"))
