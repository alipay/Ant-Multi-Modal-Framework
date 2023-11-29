import json
import csv
import time

def get_ranked_result(json_data, ranked_data, rank_top_k, group_split_list):
    len_group = len(group_split_list)
    new_count_list = []
    for i in range(len(json_data)):
        if ranked_data[i] <= rank_top_k:
            new_count_list.append(json_data[i])
    sorted_list_all = sorted(new_count_list, reverse=True)
    sorted_list = []
    for sorted_num in sorted_list_all:
        if sorted_num > 0:
            sorted_list.append(sorted_num)

    '''get_group_num_list'''
    assert sorted_list[0] >= group_split_list[0]
    assert sorted_list[-1] <= group_split_list[-2]

    rank_group_from_dict = {}

    for sorted_num in sorted_list:
        for j in range(len_group):
            if sorted_num >= group_split_list[j]:
                break
        rank_group_from_dict[sorted_num] = j

    '''获得满足条件的备选项'''
    proper_choice_list = []
    for i in range(len(json_data)):
        if json_data[i] in sorted_list:
            proper_choice_list.append(i)

    '''备选项概率分组'''
    group_num_lists = []
    for i in range(len_group):
        group_num_lists.append([])

    for i in range(len(json_data)):
        if json_data[i] in sorted_list:
            group_rank = rank_group_from_dict[json_data[i]]
            for j in range(group_rank, len_group):
                group_num_lists[j].append(i)

    return proper_choice_list, group_num_lists

def percentage_rechange(json_dir, threshold, group_split_list, rank_top_k, json_save_dir, VTM_save_dir, csv_save_dir):

    with open(json_dir, 'r') as file:
        json_data = json.load(file)

    vocab_length = len(json_data['name'])

    id_list = json_data['id']
    name_list = json_data['name']
    total_list = json_data['total_count']
    verb_list = json_data['verb_count']
    noun_list = json_data['noun_count']
    adj_list = json_data['adj_count']
    rank_list = json_data['rank']
    verb_possi_list = json_data['verb_percentage']
    noun_possi_list = json_data['noun_percentage']
    adj_possi_list = json_data['adj_percentage']

    for i in range(vocab_length):
        possi_key_list = ['verb_percentage', 'noun_percentage', 'adj_percentage']
        count_key_list = ['verb_count', 'noun_count', 'adj_count']
        actual_percent_count = 0.0
        '''计算大于threshold的总概率'''
        for key_name in possi_key_list:
            if json_data[key_name][i] >= threshold:
                actual_percent_count += json_data[key_name][i]
        '''重置概率，将小于threshold的数值直接置0'''
        for possi_name, count_name in zip(possi_key_list, count_key_list):
            if json_data[possi_name][i] >= threshold:
                json_data[possi_name][i] = json_data[possi_name][i] / actual_percent_count
            else:
                json_data[possi_name][i] = 0.0
                json_data[count_name][i] = 0

    '''处理以获得VTM的拟似输入'''
    VTM_save_dict = {}
    percentage_list = []

    shuliang = [0, 0, 0, 0, 0]

    for i in range(vocab_length):
        dict_2 = {}
        if rank_list[i] <= rank_top_k:
            possi_key_list = ['verb_percentage', 'noun_percentage', 'adj_percentage']
            for possi_name in possi_key_list:
                if json_data[possi_name][i] >= threshold:
                    dict_2[possi_name.split('_')[0]] = json_data[possi_name][i]
        percentage_list.append(dict_2)
    #     shuliang[len(dict_2)] += 1
    #     if len(dict_2) > 2:
    #         print(i)
    # print(shuliang)
    # exit(7)

    VTM_save_dict['percentage_all'] = percentage_list
    '''将数量相近的word进行分组'''
    for count_name in count_key_list:
        proper_choice_list, group_num_lists = get_ranked_result(json_data[count_name], json_data['rank'], rank_top_k, group_split_list)
        dict_tm = {
            'proper_choice_list': proper_choice_list,
            'group_num_lists': group_num_lists
        }
        c_type = '%s_choice'%(count_name.split('_')[0])
        VTM_save_dict[c_type] = dict_tm


    '''json_save for VTM'''
    with open(VTM_save_dir, "w", encoding='utf-8') as dump_f:
        json.dump(VTM_save_dict, dump_f)

    '''json_save'''
    with open(json_save_dir, "w", encoding='utf-8') as dump_f:
        json.dump(json_data, dump_f)

    csv_head = json_data.keys()
    csv_lists = [csv_head]
    for i in range(vocab_length):
        csv_line = [i, name_list[i], total_list[i], verb_list[i], noun_list[i], adj_list[i], rank_list[i],
                    verb_possi_list[i], noun_possi_list[i],
                    adj_possi_list[i]]
        csv_lists.append(csv_line)

    with open(csv_save_dir, 'w', newline="") as csv_f:
        csv_writer = csv.writer(csv_f)
        csv_writer.writerows(csv_lists)

if __name__ == '__main__':
    suffix = time.asctime()

    # '''webvid setting'''
    # father_dir = 'xxx/webvid_0719/'
    # json_dir = father_dir + 'count_ss_webvid.json'
    # percentage_chosen_threshold = 0.25
    # group_split_list = [50000, 30000, 10000, 3000, 0]
    # rank_top_k = 3000
    # json_save_dir = father_dir + 'count_reprocess_webvid.json'
    # VTM_save_dir = father_dir + 'VTM_choice_webvid.json'
    # csv_save_dir = father_dir + 'count_reprocess_webvid.csv'
    #
    # percentage_rechange(json_dir, percentage_chosen_threshold, group_split_list, rank_top_k, json_save_dir, VTM_save_dir, csv_save_dir)

    '''cocovg_setting'''
    father_dir = 'xxx/cocovg_0719/'
    json_dir = father_dir + 'count_vg_0720.json'
    percentage_chosen_threshold = 0.25
    group_split_list = [50000, 30000, 10000, 3000, 0]
    rank_top_k = 3000
    json_save_dir = father_dir + 'count_reprocess_vg_0720.json'
    VTM_save_dir = father_dir + 'VTM_choice_vg_0720.json'
    csv_save_dir = father_dir + 'count_reprocess_vg_0720.csv'

    percentage_rechange(json_dir, percentage_chosen_threshold, group_split_list, rank_top_k, json_save_dir,
                        VTM_save_dir, csv_save_dir)