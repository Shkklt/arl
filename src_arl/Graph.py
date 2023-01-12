import torch
import numpy as np
from collections import defaultdict
import copy


class Knowledge_graph():
    def __init__(self, option, data_loader, data):
        self.option = option
        self.data = data
        self.data_loader = data_loader
        self.out_array = None
        self.all_correct = None
        self.construct_graph()
        self.en_re_dict = None

    # 根据原始数据构建知识图谱，out_array存储每个节点向外的出口路径数组
    def construct_graph(self):
        all_out_dict = defaultdict(list)
        all_correct = defaultdict(set)
        for head, relation, tail in self.data:
            all_out_dict[head].append((relation, tail))
            all_correct[(head, relation)].add(tail)

        out_array = np.ones((self.option.num_entity, self.option.max_out, 2), dtype=np.int64)
        out_array[:, :, 0] *= self.data_loader.relation2num["Pad"]
        out_array[:, :, 1] *= self.data_loader.entity2num["Pad"]
        # self.en_re_dict = self.data_loader.analogy_dict
        more_out_count = 0
        for head in all_out_dict:
            out_array[head, 0, 0] = self.data_loader.relation2num["Equal"]
            out_array[head, 0, 1] = head
            num_out = 1
            for relation, tail in all_out_dict[head]:
                if num_out == self.option.max_out:
                    more_out_count += 1
                    break

                out_array[head, num_out, 0] = relation
                out_array[head, num_out, 1] = tail
                num_out += 1

            # if(en_re_dict[str(head)]!="0"):
            #     entities_list = en_re_dict[str(head)].split(" ")
            #     # print("entities_list",entities_list)
            #
            #     for entity in entities_list:
            #         if num_out == self.option.max_out:
            #             break
            #             # print('len(self.num2relation)+ relation',14+ relation)
            #         if(entity.isnumeric()):
            #             out_array[head, num_out, 0] = self.data_loader.relation2num["Analogy"]
            #             # print("entity",entity)
            #             out_array[head, num_out, 1] = int(entity)
            #             num_out += 1
        self.out_array = torch.from_numpy(out_array)
        self.all_correct = all_correct
        # print("more_out_count", more_out_count)
        if self.option.use_cuda:
            self.out_array = self.out_array.cuda()

    # 获取从图谱上current_entities的out_relations, out_entities
    def get_out(self, current_entities, start_entities, queries, answers, all_correct, step,status):

        if(status==0):
            s_para =20
        else:
            s_para=100

        ret = copy.deepcopy(self.out_array[current_entities, :, :])
        # print("len of analogy",len(self.data_loader.analogy_dict))
        for i in range(current_entities.shape[0]):
            if current_entities[i] == start_entities[i]:
                relations = ret[i, :, 0]
                entities = ret[i, :, 1]
                mask = queries[i].eq(relations) & answers[i].eq(entities)
                #mask = queries[i].eq(relations)
                ret[i, :, 0][mask] = self.data_loader.relation2num["Pad"]
                ret[i, :, 1][mask] = self.data_loader.entity2num["Pad"]

            # if(step<self.option.max_step_length - 1):
            #     relations = ret[i, :, 0]
            #
            #     analogy_relation = torch.ones_like(relations) * self.data_loader.relation2num["Analogy"]
            #     mask = analogy_relation.eq(relations)
            #
            #     ret[i, :, 0][mask] = self.data_loader.relation2num["Pad"]
            #     ret[i, :, 1][mask] = self.data_loader.entity2num["Pad"]
                # for j in range(entities.shape[0]):
                #     if relations[j] == self.data_loader.relation2num["Analogy"]:
                #         relations[j] = self.data_loader.relation2num["Pad"]
                #         entities[j] = self.data_loader.entity2num["Pad"]

            # if(step ==1):
            #     relations = ret[i, :, 0]
            #
            #     analogy_relation = torch.ones_like(relations) * self.data_loader.relation2num["Analogy"]
            #     mask = analogy_relation.eq(relations)
            #
            #     ret[i, :, 0][mask] = self.data_loader.relation2num["Analogy1"]

            if step == self.option.max_step_length - 1:
                relations = ret[i, :, 0]
                entities = ret[i, :, 1]
                answer = answers[i]
                # print("i",i)
                # print("all_correct[int(i/rollouts)]",all_correct[int(i/rollouts)])
                # print("all_correct[int(i)]",all_correct[i])

                # analogy_relation = torch.ones_like(relations) * self.data_loader.relation2num["Analogy"]
                # mask = analogy_relation.eq(relations)
                #
                # ret[i, :, 0][mask] = self.data_loader.relation2num["Analogy2"]
                # print("answer",str(answer.item()))
                # print("self.data_loader.analogy_dict.keys()",self.data_loader.analogy_dict.keys())
                # if(str(answer.item()) in self.data_loader.analogy_dict.keys()):
                #     ana_queries = True
                # else:
                #     ana_queries=False

                ana_result = self.data_loader.analogy_dict[str(answer.item())]

                entities_key = False
                for e in entities:
                    if(str(e.item()) in ana_result):
                        entities_key=True
                        break


                # if(str(start_entities[i].item()) in ana_result or entities_key):
                if(entities_key):
                    ana_queries = True
                else:
                    ana_queries=False


                # if answer not in entities and i%s_para==0:
                #     relations[0] = self.data_loader.relation2num["Analogy"]
                #     entities[0] = answer

                if ana_queries and answer not in entities and i%s_para==0:
                    relations[0] = self.data_loader.relation2num["Analogy"]
                    entities[0] = answer

                for j in range(entities.shape[0]):
                    if entities[j] in all_correct[i] and entities[j] != answer:
                        relations[j] = self.data_loader.relation2num["Pad"]
                        entities[j] = self.data_loader.entity2num["Pad"]

                        #print("step_in")


        return ret

    def get_next(self, current_entities, out_ids):
        next_out = self.out_array[current_entities, :, :]
        next_out_list = list()
        for i in range(out_ids.shape[0]):
            next_out_list.append(next_out[i, out_ids[i]])
        next_out = torch.stack(next_out_list)
        next_entities = next_out[:, 1]
        return next_entities

    def get_all_correct(self, start_entities_np, relations_np):
        all_correct = list()
        for i in range(start_entities_np.shape[0]):
            all_correct.append(self.all_correct[(start_entities_np[i], relations_np[i])])
        return all_correct