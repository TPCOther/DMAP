import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata

def print_memory_usage():
    allocated = torch.cuda.memory_allocated("cuda:0")
    reserved = torch.cuda.memory_reserved("cuda:0")
    print(f"Allocated: {allocated / (1024 ** 2):.2f} MB")
    print(f"Reserved: {reserved / (1024 ** 2):.2f} MB")

def cal_sub_len(sub_weights, substituions):
    sub_len = [0] * len(sub_weights)

    for i in range(len(sub_weights)):
        sub_len[i] = round(len(substituions[i])*sub_weights[i])
    return sub_len

def get_attention_output(model, dataset, target_layer=5, batch_size=16):
    output_score, masks, attention_output = model.get_attentions_output(dataset, target_layer, batch_size)
    output_scores = output_score
    return attention_output, masks, output_scores

def rrf_rank(att_score, probe_score):
    att_rank = rankdata(att_score, method='ordinal')
    probe_score = -probe_score
    kde_rank = rankdata(probe_score, method='ordinal')
    rrf_rank = []
    for i in range(len(att_rank)):
        rank = 1.0 / (60 + att_rank[i]) + 1.0 / (60 + kde_rank[i])
        rrf_rank.append(rank)
    final_rank = np.argsort(rrf_rank)[::-1]
    return final_rank, rrf_rank

class UCB:
    def __init__(self, num_options):
        self.num_options = num_options
        self.total_rewards = np.zeros(num_options)  # 存储每个选项的累积奖励
        self.num_selections = np.zeros(num_options)  # 存储每个选项被选择的次数
        self.cur_len = np.zeros(num_options)  # 当前长度
        self.total_selections = 0  # 总的选择次数
        self.reward_scaling_factor = 3  # 奖励缩放因子

    def select_option(self):
        # 计算每个选项的置信上界
        ucb_values = self.total_rewards / (self.num_selections + 1e-5) + np.sqrt(2 * np.log(self.total_selections + 1) / (self.num_selections + 1e-5))
        
        # 选择置信上界最大的选项
        exp_ucb_values = np.exp(ucb_values)
        selected_option = exp_ucb_values / np.sum(exp_ucb_values)

        self.cur_weights = selected_option

        for i in range(self.num_options):
            self.num_selections[i] += selected_option[i]
        self.total_selections += 1

        return selected_option[:]

    def update_rewards(self, reward):
        for i in range(self.num_options):
            self.total_rewards[i] += reward[i]*self.reward_scaling_factor

    def set_len(self, cur_len):
        self.cur_len = cur_len
    
    def get_len(self):
        return self.cur_len

class ProbeClassifier(nn.Module):
    def __init__(self):
        super(ProbeClassifier, self).__init__()
        self.dense = nn.Linear(768 * 2, 768)
        self.norm = nn.BatchNorm1d(768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
    def cal_probe_score(self, atts, true_label):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(atts)
            scores = F.softmax(outputs, dim=1)
            scores = scores[:, true_label]
        
        return scores.cpu().detach().numpy()

class DMAP:
    def __init__(self, mix_nums, probe_path, reserve, target_layer, device) -> None:
        self.ucb = UCB(mix_nums)

        self.probe = ProbeClassifier()
        self.probe.load_state_dict(torch.load(probe_path))
        self.probe.to(device)
        self.probe.eval()

        self.reserve = reserve
        self.target_layer = target_layer

        self.mask_cache = None
        self.att_cache = None
        self.order_cache = None
    
    
    def get_att_order(self, tgt_model, dataset, ori_att, ori_label):
        atts, masks, atts_cache = get_attention_output(tgt_model, dataset, self.target_layer)
        atts_score = atts[:, 0, :].squeeze(1)
        probe_score = atts_score.reshape(-1, atts_score.size(-1) * 2)
        atts_score = atts_score[::2]

        target_label = 0 if ori_label == 1 else 1
        probe_score = self.probe.cal_probe_score(probe_score, target_label)

        sim_score = []
        for att in atts_score:
            sim_score.append(F.cosine_similarity(ori_att, att.unsqueeze(0)).item())

        order, _ = rrf_rank(sim_score, probe_score)
        reorder1 = order*2
        reorder2 = order*2 + 1
        reorder = np.empty(order.size * 2, dtype=order.dtype)
        reorder[0::2] = reorder1
        reorder[1::2] = reorder2
        reorder = torch.tensor(reorder.tolist()).to(atts.device)

        self.att_cache = torch.index_select(atts_cache, 0, reorder)
        self.mask_cache = torch.index_select(masks, 0, reorder)
        self.order_cache = order
        
        return order
    
    def get_att_with_update(self, tgt_model, dataset, ori_att, ori_label):
        atts, masks = get_attention_output(tgt_model, dataset, self.target_layer)
        atts_score = atts[:, 0, :].squeeze(1)
        probe_score = atts_score.reshape(-1, atts_score.size(-1) * 2)
        atts_score = atts_score[::2]

        target_label = 0 if ori_label == 1 else 1
        probe_score = self.probe.cal_probe_score(probe_score, target_label) 

        sim_score = []
        for att in atts_score:
            sim_score.append(F.cosine_similarity(ori_att, att.unsqueeze(0)).item())

        order, rrf_score = rrf_rank(sim_score, probe_score)
        reorder1 = order*2
        reorder2 = order*2 + 1
        reorder = np.empty(order.size * 2, dtype=order.dtype)
        reorder[0::2] = reorder1
        reorder[1::2] = reorder2
        reorder = torch.tensor(reorder.tolist()).to(atts.device)
 
        self.att_cache = torch.index_select(atts, 0, reorder)
        self.mask_cache = torch.index_select(masks, 0, reorder)
        self.order_cache = order

        self.adjust_weight(np.arange(len(rrf_score)), score=rrf_score)
        
        return order
    
    def verify_logist(self, tgt_model, batch_size, ori_prob = None, ori_label = None, adjust=True):
        reserve_len = len(self.att_cache) // 2 // self.reserve
        reserve_len = reserve_len * 2
        reserve_len = max(reserve_len, 2)
        
        logits, preds = tgt_model.continue_forward(self.att_cache[:reserve_len], self.mask_cache[:reserve_len], self.target_layer, batch_size)

        if adjust:
            self.adjust_weight(self.order_cache[:reserve_len], score=[max(ori_prob) - logit[ori_label] for logit in logits])

        return logits, preds
        
    
    def get_sub_by_weight(self, substituions):
        sub = []
        sub_weights = self.ucb.select_option()
        print(sub_weights)
        sub_len = cal_sub_len(sub_weights, substituions)    
        
        for i in range(self.ucb.num_options):
            if sub_len[i] == 0:
                print("zero sub, reselect")
                sub_weights[i] = 0
                sub_weights = np.exp(sub_weights)
                sub_weights = sub_weights / np.sum(sub_weights)
                sub_len = cal_sub_len(sub_weights, substituions)
                print(sub_weights)

        print(sub_len)         

        self.ucb.set_len(sub_len)
        for i in range(self.ucb.num_options):
            sub.extend(substituions[i][:sub_len[i]])

        return sub
    
    def get_sub_by_weight_reverse(self, substituions):
        sub = []
        sub_weights = self.ucb.select_option()
        print(sub_weights)
        sub_len = cal_sub_len(sub_weights, substituions)    
        
        for i in range(self.ucb.num_options):
            if sub_len[i] == 0:
                print("zero sub, reselect")
                sub_weights[i] = 0
                sub_weights = np.exp(sub_weights)
                sub_weights = sub_weights / np.sum(sub_weights)
                sub_len = cal_sub_len(sub_weights, substituions)
                print(sub_weights)

        print(sub_len)         

        self.ucb.set_len(sub_len)
        for i in range(self.ucb.num_options):
            sub.extend(substituions[i][-sub_len[i]:])

        return sub
    
    def adjust_weight(self, order, score):
        sub_len = self.ucb.get_len()
        reward = [0] * self.ucb.num_options
        reward_nums = [0] * self.ucb.num_options
        for i in range(len(score)):
            if order[i] < sub_len[0]:
                reward[0] += score[i]
                reward_nums[0] += 1
            elif order[i] < sub_len[1] + sub_len[0]:
                reward[1] += score[i]
                reward_nums[1] += 1
            else:
                reward[2] += score[i]
                reward_nums[2] += 1
        print(reward_nums)
        self.ucb.update_rewards(reward)