import os
import json
from tqdm import tqdm
import spacy
import json
import jsonlines

class Count_Class():
    def __init__(self):
        self.count_dict = {
            "VERB":{},
            "NOUN":{},
            "ADJ":{},
            "NUM":{}
        }

    def add_key_val(self, k0, v0, num=1):
        if v0 not in self.count_dict[k0].keys():
            self.count_dict[k0][v0] = num
        else:
            self.count_dict[k0][v0] += num

    def update_same(self, CountClass):
        for k1, v1 in CountClass.items():
            for k2, v2 in v1.items():
                self.add_key_val(k1, k2, v2)

    def items(self):
        return self.count_dict.items()

    def dict_print(self):
        for k1, v1 in self.count_dict.items():
            print(k1)
            for k2, v2 in v1.items():
                print('%s-%d' % (k2, v2))
            print('\n\n')

    def get_dict(self):
        return self.count_dict

class Spacy_Processor():
    def __init__(self, mode="small"):
        if mode == "small":
            self.nlp = spacy.load("en_core_web_sm")
        elif mode == "middle":
            self.nlp = spacy.load("en_core_web_md")
        elif mode == "large":
            self.nlp = spacy.load("en_core_web_lg")
        else:
            exit('wrong type in loading NLP modules!')

        self.first_importance = ["VERB", "NOUN", "ADJ", "NUM"]
        self.second_importance = ['ADP', 'ADV', 'PROPN', 'AUX']
        self.other_pos = ['CCONJ', 'PRON', 'DET', 'PUNCT', 'SCONJ', 'PART']
        self.count_all = Count_Class()

    def get_text_tokens(self, caption):
        doc = self.nlp(caption)
        text_list = [token.text for token in doc]
        return text_list

    def process_caption_into_pos_dict(self, caption):
        save_dict = {
            "VERB": [],
            "NOUN": [],
            "ADJ": [],
            "NUM": [],
            "1st": [],
            "2nd": [],
            "others": []
        }

        doc = self.nlp(caption)
        text_list = [token.text for token in doc]
        pos_list = [token.pos_ for token in doc]
        mask_1st_list = text_list.copy()
        mask_all_list = text_list.copy()

        for i in range(len(pos_list)):
            if pos_list[i] in self.first_importance:
                save_dict[pos_list[i]].append(i)
                save_dict['1st'].append(i)
                mask_1st_list[i] = pos_list[i]
                mask_all_list[i] = '[MASK]'
            elif pos_list[i] in self.second_importance:
                save_dict['2nd'].append(i)
            else:
                save_dict['others'].append(i)

        return save_dict, text_list, mask_1st_list, mask_all_list

    def count_words_belong_to_what(self, caption):
        count_dict = Count_Class()

        doc = self.nlp(caption)
        pos_list = [token.pos_ for token in doc]
        tokens_list = [token for token in doc]
        for token_l, pos_l in zip(tokens_list, pos_list):
            token_l = str(token_l)
            if pos_l in self.first_importance:
                count_dict.add_key_val(pos_l, token_l)
        self.count_all.update_same(count_dict)

        return count_dict

    def dict_save_into_json(self, dict_info, save_dir):
        with open(save_dir, "w", encoding='utf-8') as dump_f:
            json.dump(dict_info, dump_f)


if __name__ == '__main__':
    father_dir = 'xxx/ssm_webvid/anno_files/'
    train_file_list = ['train.jsonl']
    final_save_path = os.path.join(father_dir, 'webvid_initial.json')

    SPC = Spacy_Processor()

    idx = 0

    for train_file in train_file_list:
        train_file_path = os.path.join(father_dir, train_file)
        print("begin to process %s"%train_file)
        with jsonlines.open(train_file_path) as reader:
            for obj in tqdm(reader):
                idx += 1
                caption = obj['caption']

                count_dict = SPC.count_words_belong_to_what(caption)

                if idx%10000 == 0:
                    print("idx %07d save success!"%idx)
                    cur_save_path = final_save_path[:-5]+("_%07d.json"%(idx))
                    SPC.dict_save_into_json(SPC.count_all.get_dict(), cur_save_path)

    SPC.dict_save_into_json(SPC.count_all.get_dict(), final_save_path)
