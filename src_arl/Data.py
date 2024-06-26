import os
from collections import defaultdict
import numpy as np
import copy
import json

class Data_loader():
    def __init__(self, option):
        self.option = option
        self.include_reverse = True

        self.train_data = None
        self.test_data = None
        self.valid_data = None
        self.train_graph_data =None

        self.entity2num = None
        self.num2entity = None

        self.relation2num = None
        self.num2relation = None
        self.relation2inv = None

        self.analogy_data=None
        self.analogy_dict=None

        self.num_relation = 0
        self.num_entity = 0
        self.num_operator = 0

        data_path = os.path.join(self.option.datadir, self.option.dataset)
        self.load_data_all(data_path)

    def load_data_all(self, path):
        train_data_path = os.path.join(path, "train.txt")
        test_data_path = os.path.join(path, "test.txt")
        valid_data_path = os.path.join(path, "valid.txt")
        entity_path = os.path.join(path, "entity_vocab.json")
        relations_path = os.path.join(path, "relation_vocab.json")
        graph_data = os.path.join(path, "graph.txt")
        analogy_data_path = os.path.join(path, "analogy.txt")
        analogy_dict_path = os.path.join(path, "analogy.json")

        # self.entity2num, self.num2entity = self._load_dict(entity_path)
        # self.relation2num, self.num2relation = self._load_dict(relations_path)


        # self._augment_reverse_relation()

        self.entity2num, self.num2entity = self._load_vocab(entity_path)
        self.relation2num, self.num2relation = self._load_vocab(relations_path)
        for i in range(61):
            self._add_item(self.relation2num, self.num2relation, "analogy"+str(i))
        self._add_item(self.relation2num, self.num2relation, "Analogy")
        # self._add_item(self.relation2num, self.num2relation, "Analogy2")
        # self._add_item(self.relation2num, self.num2relation, "Equal")
        # self._add_item(self.relation2num, self.num2relation, "Pad")
        # self._add_item(self.relation2num, self.num2relation, "Start")
        # self._add_item(self.entity2num, self.num2entity, "Pad")
        print(self.relation2num)

        self.num_relation = len(self.relation2num)
        self.num_entity = len(self.entity2num)
        print("num_relation", self.num_relation)
        print("num_entity", self.num_entity)

        self.train_data = self._load_data(train_data_path)
        self.valid_data = self._load_data(valid_data_path)
        self.test_data = self._load_data(test_data_path)
        self.train_graph_data = self._load_ddd(graph_data)
        self.analogy_dict = self._load_json(analogy_dict_path)
        print("self.analogy_dict",len(self.analogy_dict))
        # print("self.analogy_dict",self.analogy_dict.keys())
        self.analogy_data = self._load_ana(analogy_data_path)

    def _load_data(self, path):
        data = [l.strip().split("\t") for l in open(path, "r").readlines()]
        triplets = list()
        for item in data:
            head = self.entity2num[item[0]]
            tail = self.entity2num[item[2]]
            relation = self.relation2num[item[1]]
            triplets.append([head, relation, tail])
            # if self.include_reverse:
            #     inv_relation = self.relation2num["inv_" + item[1]]
            #     triplets.append([tail, inv_relation, head])
        return triplets

    def _load_ddd(self, path):
        data = [l.strip().split("\t") for l in open(path, "r").readlines()]
        triplets = list()
        for item in data:
            head = self.entity2num[item[0]]
            tail = self.entity2num[item[2]]
            relation = self.relation2num[item[1]]
            triplets.append([head, relation, tail])
        return triplets

    def _load_dict(self, path):
        obj2num = defaultdict(int)
        num2obj = defaultdict(str)
        data = [l.strip() for l in open(path, "r").readlines()]
        for num, obj in enumerate(data):
            obj2num[obj] = num
            num2obj[num] = obj
        return obj2num, num2obj

    def _load_vocab(self,path):
        data = json.load(open(path))

        obj2num = defaultdict(int)
        num2obj = defaultdict(str)

        # self.rev_relation_vocab = dict([(v, k) for k, v in relation_vocab.items()])
        # self.rev_entity_vocab = dict([(v, k) for k, v in entity_vocab.items()])
        # print("data",data)
        for num, obj in enumerate(data):
            obj2num[obj] = data[obj]
            num2obj[data[obj]] = obj
        return obj2num, num2obj
    def _load_json(self,path):
        data = json.load(open(path))
        obj2num = defaultdict(str)
        # self.rev_relation_vocab = dict([(v, k) for k, v in relation_vocab.items()])
        # self.rev_entity_vocab = dict([(v, k) for k, v in entity_vocab.items()])
        # print("data",data)
        for num, obj in enumerate(data):
            obj2num[obj] = data[obj]
        return obj2num

    def _load_ana(self,path):
        data = [l.strip().split(",") for l in open(path, "r").readlines()]
        triplets = list()

        for item in data:

            head = int(item[0].strip('['))
            tail = int(item[2].strip(']'))

            r = "analogy"+str(item[1].strip())
            relation = self.relation2num[r]
            triplets.append([head, relation, tail])
        return triplets

    def _augment_reverse_relation(self):
        num_relation = len(self.num2relation)
        temp = list(self.num2relation.items())
        self.relation2inv = defaultdict(int)
        for n, r in temp:
            rel = "_" + r
            num = num_relation + n
            self.relation2num[rel] = num
            self.num2relation[num] = rel
            self.relation2inv[n] = num
            self.relation2inv[num] = n

    def _add_item(self, obj2num, num2obj, item):
        count = len(obj2num)
        obj2num[item] = count
        num2obj[count] = item

    def get_train_graph_data(self):
        with open(os.path.join(self.option.this_expsdir, "train_log.txt"), "a+", encoding='UTF-8') as f:
            f.write("Train graph contains " + str(len(self.train_data)) + " triples\n")
        # return np.vstack((np.array(self.train_graph_data, dtype=np.int64),np.array(self.analogy_data, dtype=np.int64)))
        return np.array(self.train_graph_data, dtype=np.int64)

    def get_train_data(self):
        with open(os.path.join(self.option.this_expsdir, "train_log.txt"), "a+", encoding='UTF-8') as f:
            f.write("Train data contains " + str(len(self.train_graph_data)) + " triples\n")
        return np.array(self.train_data, dtype=np.int64)

    def get_test_graph_data(self):
        with open(os.path.join(self.option.this_expsdir, "test_log.txt"), "a+", encoding='UTF-8') as f:
            f.write("Test graph contains " + str(len(self.train_graph_data)) + " triples\n")
        return np.array(self.train_graph_data, dtype=np.int64)

    def get_test_data(self):
        with open(os.path.join(self.option.this_expsdir, "test_log.txt"), "a+", encoding='UTF-8') as f:
            f.write("Test graph contains " + str(len(self.test_data)) + " triples\n")
        return np.array(self.test_data, dtype=np.int64)

    def get_analogy_data(self):
        return self.analogy_data