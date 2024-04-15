import json
region_descriptions = json.load(open("/Users/xuan/Downloads/visual_genome/region_descriptions.json", "r"))

merge_result = []
for regions in region_descriptions:
    for region in regions['regions']:
        result_dict = {}
        result_dict['image_id'] = "{}.jpg".format(region['image_id'])
        result_dict['sentence'] = region['phrase']
        bbox = [region['x'], region['y'], region['x']+region['width'], region['y']+region['height']]
        result_dict['bbox'] = bbox
        merge_result.append(result_dict)

print(len(merge_result))
json.dump(merge_result, open("/Users/xuan/Downloads/visual_genome/rd_vg.json", "w"))
