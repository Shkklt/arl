import json
import csv
import argparse
from collections import defaultdict
import os
import numpy as np

import itertools



def _load_ddd(path,entity2num,relation2num):
    data = [l.strip().split("\t") for l in open(path, "r").readlines()]
    # triplets = list()
    triplets = list()
    for item in data:
        head = entity2num[item[0]]
        tail = entity2num[item[2]]
        relation = relation2num[item[1]]
        triplets.append([head, relation, tail])
    return triplets

def _load_vocab(path):
    data = json.load(open(path))

    obj2num = defaultdict(int)
    num2obj = defaultdict(str)


    for num, obj in enumerate(data):
        obj2num[obj] = data[obj]
        num2obj[data[obj]] = obj
    return obj2num, num2obj

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

def analogy_entity_b(all_out_array_dict):
    en_re_dict = defaultdict(int)
    list_duplicate =[]

    list_relation = []
    count=0
    # print("len(all_out_array_dict.values())",len(all_out_array_dict.values()))
    for relation_list in all_out_array_dict.values():

        count=count+1
        if(count % 2)==0:
            continue
        list_relation.append(relation_list)



    # print(len(list_relation))

    # print('len(self.all_out_array_dict.values()',len(self.all_out_array_dict.values()))
    count=0
    # print("len(all_out_array_dict.values())",len(all_out_array_dict.values()))
    for relation_list in all_out_array_dict.values():

        count=count+1
        if(count % 2)==0:
            continue

        # print(relation_list)
        # r = list(relation_list)
        # r.sort()
        str = " ".join('%s' %id for id in relation_list)

        relation_num = len(relation_list)

        # print("relation_list",relation_list)
        # print("relation_num",relation_num)



        if(str not in list_duplicate and str!="" and relation_num>1):

            list_duplicate.append(str)
            key_list = get_keys(all_out_array_dict,relation_list)
            # print('len(key_list)',len(key_list))


            if(len(key_list)>5000):

                print("relation_list",relation_list)
                print("tmp_key length",len(key_list))
                tmp_list=[]
                for m,n in key_list:
                    tmp_list.append(m)

                # count=0
                tmp_key = tmp_list.copy()

                # print("tmp_list",tmp_list)
                for key in tmp_list:
                    # if(key==1500):
                    #     print("tmp_list",tmp_list)
                    # print('key',key)
                    # if(key==21500):
                    #     print("tmp_list",tmp_list)
                    tmp_key.remove(key)

                    # if(key==1500):
                    #     print("tmp_key",tmp_key)
                    # if(key==21500):
                    #     print("tmp_key",tmp_key)

                    #print(len(tmp_key))
                    en_re_dict[key]=(" ".join('%s' %id for id in tmp_key))

                    tmp_key = tmp_list.copy()
                    # if(count>200):
                    #     break
    # lt = en_re_dict.keys()
    # print('len(en_re_dict.keys())',len(en_re_dict.keys()))
    # print('all_out_array_dict[1500,0]',self.all_out_array_dict[1500,0])
    return en_re_dict

def analogy_entity(all_out_array_dict):
    list_ana = list()
    list_duplicate =[]

    list_relation = []
    count=0
    # print("len(all_out_array_dict.values())",len(all_out_array_dict.values()))
    for relation_list in all_out_array_dict.values():

        count=count+1
        if(count % 2)==0:
            continue
        list_relation.append(relation_list)



    # print(len(list_relation))

    # print('len(self.all_out_array_dict.values()',len(self.all_out_array_dict.values()))
    count=0
    count11 = 0
    # print("len(all_out_array_dict.values())",len(all_out_array_dict.values()))
    for relation_list in all_out_array_dict.values():

        count=count+1
        if(count % 2)==0:
            continue

        # print(relation_list)
        # r = list(relation_list)
        # r.sort()
        str1 = " ".join('%s' %id for id in relation_list)

        relation_num = len(relation_list)

        # print("relation_list",relation_list)
        # print("relation_num",relation_num)



        if(str1 not in list_duplicate and str1!="" and relation_num>1):

            list_duplicate.append(str1)
            key_list = get_keys(all_out_array_dict,relation_list)
            # print('len(key_list)',len(key_list))


            if(len(key_list)>1 and len(key_list)<350):


                print("relation_list",relation_list)
                print("tmp_key length",len(key_list))
                tmp_list=[]
                for m,n in key_list:
                    tmp_list.append(m)

                new_list_ana = (list(itertools.combinations(list(tmp_list),2)))

                for l in new_list_ana:
                    # print(l[0],l[1])
                    # print("l.insert(analogy+str(count))",[l[0],"analogy"+str(count),l[1]])
                    list_ana.append([l[0],count11,l[1]])

                count11+=1

    print("count11",count11)

    return list_ana

def main():
    root_dir = '../'
    dir = root_dir+'datasets/WN18RRF/'

    train_data_path = os.path.join(dir, "train.txt")
    test_data_path = os.path.join(dir, "test.txt")


    entity_path = os.path.join(dir, "entity_vocab.json")
    relations_path = os.path.join(dir, "relation_vocab.json")
    entity2num, num2entity = _load_vocab(entity_path)
    relation2num, num2relation = _load_vocab(relations_path)

    data1 = _load_ddd(train_data_path,entity2num,relation2num)
    data2 = _load_ddd(test_data_path,entity2num,relation2num)

    data = data1

    print(len(data))

    all_out_array_dict = defaultdict(set)

    for head,relation,tail in data:
        all_out_array_dict[head,0].add(relation)
        all_out_array_dict[head,1].add(tail)




    list_ana = analogy_entity(all_out_array_dict)
    output_analogy_file1 = os.path.join(dir, "analogy.txt")

    print("len(list_ana)",len(list_ana))

    file = open(output_analogy_file1,'w')
    for i in list_ana:
        file.writelines(str(i)+'\n')
    file.close()
    print("DONE for analogy.txt")

    en_re_dict = analogy_entity_b(all_out_array_dict)
    print("len(en_re_dict)",len(en_re_dict))
    # print("len(en_re_dict)",en_re_dict)

    output_analogy_file2 = os.path.join(dir, "analogy.json")
    with open(output_analogy_file2, 'w') as fout:
        json.dump(en_re_dict, fout)


    print("DONE for analogy.json")



if __name__ == "__main__":
    main()