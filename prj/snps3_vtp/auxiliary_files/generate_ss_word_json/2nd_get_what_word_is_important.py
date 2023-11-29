import json
import os
import time
import csv
from tqdm import tqdm

def get_json_list(json_dir):
    with open(json_dir, 'r') as file:
        vocab_data = json.load(file)
    return vocab_data

def get_idx_in_list(ini_list, num):
    for i in range(len(ini_list)):
        if ini_list[i] == num:
            return i
    exit('oop!')

def organize_dict(vocab_data, dict_dir, json_save_dir, csv_save_dir, abandon_list=None):
    vocab_length = len(vocab_data)
    vocab_dict = {}
    id_list = []
    name_list = vocab_data
    verb_list = []
    noun_list = []
    adj_list = []
    total_list = []
    rank_list = []
    verb_possi_list = []
    noun_possi_list = []
    adj_possi_list = []
    for i in range(len(vocab_data)):
        vocab_dict[vocab_data[i]] = i
        id_list.append(i)
        verb_list.append(0)
        noun_list.append(0)
        adj_list.append(0)
        total_list.append(0)
        rank_list.append(0)
        verb_possi_list.append(0.0)
        noun_possi_list.append(0.0)
        adj_possi_list.append(0.0)

    iter_cur = 0

    dict_info = get_json_list(dict_dir)
    print('--' * 30)
    print(dict_dir)
    for key1, val1 in dict_info.items():
        print('*' * 30)
        print('begin to process %s' % key1)
        if key1 == "NUM":
            continue
        for key2, val2 in val1.items():
            iter_cur += 1
            if iter_cur % 10000 == 0:
                print('%d times!' % iter_cur)
            # if iter_cur >= 10000:
            #     break
            if key2 not in name_list:
                continue
            if key2 in abandon_list or len(key2) == 1:
                continue
            else:
                idx = vocab_dict[key2]
                if key1 == 'VERB':
                    verb_list[idx] += val2
                elif key1 == 'NOUN':
                    noun_list[idx] += val2
                elif key1 == 'ADJ':
                    adj_list[idx] += val2
                else:
                    exit('wrong key command!')
                total_list[idx] += val2


    '''count rank'''
    ranked_total_num = sorted(total_list, reverse=True)
    for i in range(vocab_length):
        rank_i = get_idx_in_list(ranked_total_num, total_list[i])
        rank_list[i] = rank_i

    '''count percentage'''
    for i in range(vocab_length):
        total_float = float(total_list[i])
        total_float = max(0.1, total_float)
        verb_possi_list[i] = float(verb_list[i])/total_float
        noun_possi_list[i] = float(noun_list[i])/total_float
        adj_possi_list[i] = float(adj_list[i])/total_float
        assert verb_list[i] + noun_list[i] + adj_list[i] == total_list[i]


    for i in range(vocab_length):
        if total_list[i] > 1000000:
            print('id:%d-%s(T%d)\tTotal:%d\t || Verb:%d(%.1f%%)\tAdj:%d(%.1f%%)\tNum:%d(%.1f%%)\t'%
                  (i, name_list[i], rank_list[i], total_list[i], verb_list[i], verb_possi_list[i]*100, noun_list[i], noun_possi_list[i]*100,
                   adj_list[i], adj_possi_list[i]*100))

    json_save_dict = {
        'id': id_list,
        'name': name_list,
        'total_count': total_list,
        'verb_count': verb_list,
        'noun_count': noun_list,
        'adj_count': adj_list,
        'rank': rank_list,
        'verb_percentage': verb_possi_list,
        'noun_percentage': noun_possi_list,
        'adj_percentage': adj_possi_list,
    }

    with open(json_save_dir, "w", encoding='utf-8') as dump_f:
        json.dump(json_save_dict, dump_f)

    '''csv save'''
    csv_head = json_save_dict.keys()
    csv_lists = [csv_head]
    for i in range(vocab_length):
        csv_line = [i, name_list[i], total_list[i], verb_list[i], noun_list[i], adj_list[i], rank_list[i], verb_possi_list[i], noun_possi_list[i],
                    adj_possi_list[i]]
        csv_lists.append(csv_line)

    with open(csv_save_dir, 'w', newline="") as csv_f:
        csv_writer = csv.writer(csv_f)
        csv_writer.writerows(csv_lists)


    '''     dict info
    dict_keys(['VERB', 'NOUN', 'ADJ', 'NUM'])
    '''

if __name__ == '__main__':
    suffix = time.asctime()

    # '''webvid setting'''
    # father_dir = 'xxx/webvid_0719/'
    # vocab_json_dir = father_dir + 'BERT_vocabulary.json'
    # dict_file = father_dir + 'webvid_initial.json'
    # json_save_dir = father_dir + 'count_ss_webvid.json'
    # csv_save_dir = father_dir + 'count_ss_webvid.csv'
    # abandon_list = ["background", "backgrounds", "be", "is", "was", "were", "are"]
    #
    # vocab_data = get_json_list(vocab_json_dir)
    #
    # organize_dict(vocab_data, dict_file, json_save_dir, csv_save_dir, abandon_list)

    '''cocovg_setting'''
    father_dir = 'xxx/cocovg_0719/'
    vocab_json_dir = father_dir + 'token.json'
    dict_file = father_dir + 'cocovg_word_count_dict.json'
    json_save_dir = father_dir + 'count_vg_0720.json'
    csv_save_dir = father_dir + 'count_vg_0720.csv'
    abandon_list = ["be", "is", "was", "were", "are"]

    vocab_data = get_json_list(vocab_json_dir)

    organize_dict(vocab_data, dict_file, json_save_dir, csv_save_dir, abandon_list)