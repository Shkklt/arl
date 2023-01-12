import torch.nn as nn
import numpy as np
import torch
import logging as log
from collections import defaultdict


class Policy_step(nn.Module):
    def __init__(self, option):
        super(Policy_step, self).__init__()
        self.option = option
        self.lstm_cell = torch.nn.LSTMCell(input_size=self.option.action_embed_size,
                          hidden_size=self.option.state_embed_size)
    '''
    - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
    - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
    - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
    - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
    '''
    def forward(self, prev_action, prev_state):
        output, new_state = self.lstm_cell(prev_action, prev_state)
        return output, (output, new_state)

class Policy_mlp(nn.Module):
    def __init__(self, option):
        super(Policy_mlp, self).__init__()
        self.option = option
        self.hidden_size = option.mlp_hidden_size
        self.mlp_l1= nn.Linear(self.option.state_embed_size + self.option.action_embed_size,
                               self.hidden_size, bias=True)
        self.mlp_l2 = nn.Linear(self.hidden_size, self.option.action_embed_size, bias=True)

    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = torch.relu(self.mlp_l2(hidden)).unsqueeze(1)
        return output


class Agent(nn.Module):
    def __init__(self, option, data_loader, graph=None):
        super(Agent, self).__init__()
        self.option = option
        self.data_loader = data_loader
        self.graph = graph
        self.relation_embedding = nn.Embedding(self.option.num_relation, self.option.relation_embed_size)
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)
        self.policy_step = Policy_step(self.option)
        self.policy_mlp = Policy_mlp(self.option)

        if self.option.use_entity_embed:
            self.entity_embedding = nn.Embedding(self.option.num_entity, self.option.entity_embed_size)
            torch.nn.init.xavier_uniform_(self.entity_embedding.weight)

    def step(self, prev_state, prev_relation, current_entities, start_entities, queries, answers, all_correct, step):

        if self.option.use_entity_embed:
            prev_action_embedding = torch.cat((self.relation_embedding(prev_relation),self.entity_embedding(current_entities)),-1)
        else:
            prev_action_embedding = self.relation_embedding(prev_relation)

        output, new_state = self.policy_step(prev_action_embedding, prev_state)

        actions_id = self.graph.get_out(current_entities, start_entities, queries, answers, all_correct, step,0)
        out_relations_id = actions_id[:, :, 0]
        out_entities_id = actions_id[:, :, 1]

        # print(out_relations_id)

        # print("out_relations_id",out_relations_id[0])

        out_relations = self.relation_embedding(out_relations_id)

        if self.option.use_entity_embed:
            out_entities = self.entity_embedding(out_entities_id)
            action = torch.cat((out_relations,out_entities),-1)
            # print('action', action.shape)
            # print('out_relations',out_relations.shape)
            # print('out_entities', out_entities.shape)
            entity_embeddings= self.entity_embedding(current_entities)
            # print('current_entities',current_entities.shape)
            state = torch.cat((output.squeeze(), entity_embeddings), 1)
        else:
            state = output.squeeze()
            action= out_relations
        # next_entities = self.entity_embedding(out_entities_id)
        # print('next_entites1.shape',next_entities.shape)
        #next_entites.shape torch.Size([5120])
        # current_state = output.squeeze()
        queries_embedding = self.relation_embedding(queries)
        state_query = torch.cat([state, queries_embedding], -1)
        output = self.policy_mlp(state_query)

        prelim_scores = torch.sum(torch.mul(output, action), dim=-1)

        # print("prelim_scores",prelim_scores[0])
        # print("out_relations_id",out_relations_id[0])
        dummy_relations_id = torch.ones_like(out_relations_id, dtype=torch.int64) * self.data_loader.relation2num["Pad"]
        mask = torch.eq(out_relations_id, dummy_relations_id)
        dummy_scores = torch.ones_like(prelim_scores) * (-99999.0)
        scores = torch.where(mask, dummy_scores, prelim_scores)
        # print("scores[0]",scores[0])

        # action_prob = torch.softmax(scores, dim=1)
        # print("action_prob",action_prob)
        # action_prob2 = torch.softmax(scores, dim=1)
        # print("action_prob2",action_prob2)

        action_prob = torch.softmax(scores,dim=1)
        # action_prob =torch.argmax
        #
        action_id = torch.multinomial(action_prob, 1)


        chosen_relation = torch.gather(out_relations_id, dim=1, index=action_id).squeeze()
        next_entities = torch.gather(out_entities_id,dim=1,index=action_id).squeeze()


        logits = torch.nn.functional.log_softmax(scores, dim=1)

        # print("logits.shape",logits.shape)

        one_hot = torch.zeros_like(logits).scatter(1, action_id, 1)
        # print('one_hot',one_hot)
        # print('one_hot.shape',one_hot.shape)

        loss = -torch.sum(torch.mul(logits, one_hot), dim=1)

        # print('logits.shape',logits.shape)
        # print('loss.shape',loss.shape)

        action_id = action_id.squeeze()
        # next_entities = self.graph.get_next(current_entities, action_id)

        # print('next_entites2.shape',next_entities.shape)

        sss = self.data_loader.num2relation[(int)(queries[0])] + "\t" + self.data_loader.num2relation[(int)(chosen_relation[0])]
        #log.info(sss)

        return loss, new_state, logits, action_id, next_entities, chosen_relation

    def test_step(self, prev_state, prev_relation, current_entities, log_current_prob,
                  start_entities, queries, answers, all_correct, batch_size, step):
        if self.option.use_entity_embed:
            prev_action_embedding = torch.cat((self.relation_embedding(prev_relation),self.entity_embedding(current_entities)),-1)
        else:
            prev_action_embedding = self.relation_embedding(prev_relation)
        output, new_state = self.policy_step(prev_action_embedding, prev_state)

        actions_id = self.graph.get_out(current_entities, start_entities, queries, answers, all_correct, step,1)
        out_relations_id = actions_id[:, :, 0]
        out_entities_id = actions_id[:, :, 1]
        out_relations = self.relation_embedding(out_relations_id)


        if self.option.use_entity_embed:
            out_entities = self.entity_embedding(out_entities_id)
            action = torch.cat((out_relations,out_entities),-1)
            # print('action', action.shape)
            # print('out_relations',out_relations.shape)
            # print('out_entities', out_entities.shape)
            entity_embeddings= self.entity_embedding(current_entities)
            # print('current_entities',current_entities.shape)
            state = torch.cat((output.squeeze(), entity_embeddings), 1)
        else:
            state = output.squeeze()
            action= out_relations

        queries_embedding = self.relation_embedding(queries)
        state_query = torch.cat([state, queries_embedding], -1)
        output = self.policy_mlp(state_query)

        prelim_scores = torch.sum(torch.mul(output, action), dim=-1)
        dummy_relations_id = torch.ones_like(out_relations_id, dtype=torch.int64) * self.data_loader.relation2num["Pad"]
        mask = torch.eq(out_relations_id, dummy_relations_id)
        dummy_scores = torch.ones_like(prelim_scores) * (-99999.0)
        scores = torch.where(mask, dummy_scores, prelim_scores)

        action_prob = torch.softmax(scores, dim=1)
        log_action_prob = torch.log(action_prob)

        chosen_state, chosen_relation, chosen_entities, log_current_prob = self.test_search\
            (new_state, log_current_prob, log_action_prob, out_relations_id, out_entities_id, batch_size)

        return chosen_state, chosen_relation, chosen_entities, log_current_prob

    def test_search(self, new_state, log_current_prob, log_action_prob, out_relations_id, out_entities_id, batch_size):
        log_current_prob = log_current_prob.repeat_interleave(self.option.max_out).view(batch_size, -1)
        log_action_prob = log_action_prob.view(batch_size, -1)
        log_trail_prob = torch.add(log_action_prob, log_current_prob)
        top_k_log_prob, top_k_action_id = torch.topk(log_trail_prob, self.option.test_times)

        new_state_0 = new_state[0].repeat_interleave(self.option.max_out)\
            .view(batch_size, -1, self.option.state_embed_size)
        new_state_1 = new_state[1].repeat_interleave(self.option.max_out) \
            .view(batch_size, -1, self.option.state_embed_size)

        out_relations_id = out_relations_id.view(batch_size, -1)
        out_entities_id = out_entities_id.view(batch_size, -1)

        chosen_relation = torch.gather(out_relations_id, dim=1, index=top_k_action_id).view(-1)
        chosen_entities = torch.gather(out_entities_id, dim=1, index=top_k_action_id).view(-1)
        log_current_prob = torch.gather(log_trail_prob, dim=1, index=top_k_action_id).view(-1)

        top_k_action_id_state = top_k_action_id.unsqueeze(2).repeat(1, 1, self.option.state_embed_size)
        chosen_state = \
            (torch.gather(new_state_0, dim=1, index=top_k_action_id_state).view(-1, self.option.state_embed_size),
             torch.gather(new_state_1, dim=1, index=top_k_action_id_state).view(-1, self.option.state_embed_size))

        return chosen_state, chosen_relation, chosen_entities, log_current_prob

    def set_graph(self, graph):
        self.graph = graph

    def get_dummy_start_relation(self, batch_size):
        dummy_start_item = self.data_loader.relation2num["Start"]
        dummy_start = torch.ones(batch_size, dtype=torch.int64) * dummy_start_item
        return dummy_start

    def get_reward(self, current_entities, answers, all_correct, positive_reward, negative_reward,relations):
        # reward = (current_entities == answers).cpu()
        #
        # reward = reward.numpy()
        # condlist = [reward == True, reward == False]
        # choicelist = [positive_reward, negative_reward]
        # reward = np.select(condlist, choicelist)

        new_list =[]

        for i in range(current_entities.shape[0]):
            if(current_entities[i]==answers[i] and relations[i]>25):
                new_list.append(1)
            elif(current_entities[i]!=answers[i]):
                new_list.append(0)
            else:
                new_list.append(1)

        reward = np.array(new_list)

        return reward

    def print_parameter(self):
        for param in self.named_parameters():
            print(param[0], param[1])