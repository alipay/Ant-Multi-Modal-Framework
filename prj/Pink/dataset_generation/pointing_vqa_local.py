import json

split_id = json.load(open("/Users/xuan/Downloads/pointingqa-main/Datasets/LocalQA/localqa_testfinal_imgs.json", "r"))
print(len(split_id))
dataset_dict = json.load(open("/Users/xuan/Downloads/pointingqa-main/Datasets/LocalQA/localqa_dataset.json", "r"))

merge_dict = []
for img_id in split_id:
    for split in dataset_dict[img_id]:
        for i in range(len(split['all_objs'])):
            # assert len(split['all_ans']) == len(split['all_objs']) and len(split['all_ans']) == len(split['points']), print(split)
            qa_dict = {}
            qa_dict['id'] = split['id']
            qa_dict['question'] = split['question']
            qa_dict['all_objs'] = split['all_objs'][i]
            qa_dict['points'] = split['points'][i]
            merge_dict.append(qa_dict)
print(len(merge_dict))

json.dump(merge_dict, open("/Users/xuan/Downloads/pointingqa-main/Datasets/LocalQA/testfinal_split.json", "w"))
