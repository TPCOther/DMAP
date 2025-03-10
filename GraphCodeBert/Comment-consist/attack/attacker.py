import os
import pickle
import sys

sys.path.append('../../../')
sys.path.append('../../../python_parser')

import copy
import json
import torch
import random
import math
import operator
import numpy as np
import pandas as pd
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import select_parents, crossover, map_chromesome, mutate, is_valid_variable_name, _tokenize, \
    get_identifier_posistions_from_code, get_masked_code_by_position, get_replaced_var_code_with_meaningless_char, \
    CodePairDataset, remove_comments_and_docstrings, is_valid_identifier, get_code_tokens, isUID

from python_parser.run_parser import get_identifiers, extract_dataflow, get_example
from transformers import (RobertaForMaskedLM, RobertaTokenizer)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
             input_tokens_1,
             input_ids_1,
             position_idx_1,
             dfg_to_code_1,
             dfg_to_dfg_1,
             input_tokens_2,
             input_ids_2,
             position_idx_2,
             dfg_to_code_2,
             dfg_to_dfg_2,
             label,
             url1,
             url2

    ):
        #The first code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1=position_idx_1
        self.dfg_to_code_1=dfg_to_code_1
        self.dfg_to_dfg_1=dfg_to_dfg_1
        
        #The second code function
        self.input_tokens_2 = input_tokens_2
        self.input_ids_2 = input_ids_2
        self.position_idx_2=position_idx_2
        self.dfg_to_code_2=dfg_to_code_2
        self.dfg_to_dfg_2=dfg_to_dfg_2
        
        #label
        self.label=label
        self.url1=url1
        self.url2=url2

def compute_fitness(chromesome, code_2, codebert_tgt, tokenizer_tgt, orig_prob, orig_label, true_label , code_1, names_positions_dict, args):
    # 计算fitness function.
    # words + chromesome + orig_label + current_prob
    temp_replace = map_chromesome(chromesome, code_1, "java")
    new_feature = convert_code_to_features(temp_replace, 
                                                code_2,
                                                tokenizer_tgt,
                                                true_label,
                                                args)
    

    new_dataset = CodePairDataset([new_feature], args)
    new_logits, preds = codebert_tgt.get_results(new_dataset, args.eval_batch_size)
    # 计算fitness function
    fitness_value = orig_prob - new_logits[0][orig_label]
    return fitness_value, preds[0]

def convert_code_to_features(code1, code2, tokenizer, label, args):
    # 这里要被修改..
    feat = []
    for i, code in enumerate([code1, code2]):
        dfg, index_table, code_tokens = extract_dataflow(code, "java")

        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]

        code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg = dfg[:args.code_length+args.data_flow_length-len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        source_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args.code_length+args.data_flow_length-len(source_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        source_ids+=[tokenizer.pad_token_id]*padding_length

        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]
        feat.append((source_tokens,source_ids,position_idx,dfg_to_code,dfg_to_dfg))

    source_tokens_1,source_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1=feat[0]   
    source_tokens_2,source_ids_2,position_idx_2,dfg_to_code_2,dfg_to_dfg_2=feat[1]   
    return InputFeatures(source_tokens_1,source_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1,
                   source_tokens_2,source_ids_2,position_idx_2,dfg_to_code_2,dfg_to_dfg_2,
                     label, 0, 0)

def get_importance_score(args, example, code, code_2, words_list: list, variable_names: list, tgt_model, tokenizer, label_list, batch_size=16, max_length=512, model_type='classification'):
    '''Compute the importance score of each variable'''
    # label: example[1] tensor(1)
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None

    new_example = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.
    
    # code2_tokens, _, _ = _tokenize(code_2, tokenizer)

    for index, code1_tokens in enumerate([words_list] + masked_token_list):
        new_feature = convert_code_to_features(' '.join(code1_tokens),code_2,tokenizer, example[6].item(), args)
        new_example.append(new_feature)

    new_dataset = CodePairDataset(new_example, args)

    # 3. 将他们转化成features
    logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
    orig_probs = logits[0]
    orig_label = preds[0]
    # 第一个是original code的数据.
    
    orig_prob = max(orig_probs)
    # predicted label对应的probability

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])

    return importance_score, replace_token_positions, positions

def get_statement_identifier(first_idx, identifiers):
    file_path = "../dataset/data.csv"
    df = pd.read_csv(file_path, encoding="utf-8")

    row = df.loc[df['idx'] == first_idx].squeeze()
    if row.empty:
        return {}

    statement_types = ["Method_statement", "Return_statement", "If_statement",
                       "Throw_statement", "Try_statement", "For_statement"]
    statement_dict = {}

    def clean_and_filter(statement):
        cleaned = statement[1:-1].replace(" ", "").split(",")
        return [var for var in cleaned if var in identifiers]

    for stmt_type in statement_types:
        
        filtered_identifiers = clean_and_filter(row[stmt_type])
        if filtered_identifiers:
            statement_dict[stmt_type.split('_')[0]] = filtered_identifiers

    # Identify the "Other" statements
    value_list = [i for p in statement_dict.values() for i in p]
    other_statements = [var for var in identifiers if var not in value_list]
    if other_statements:
        statement_dict["Other"] = other_statements

    return statement_dict

class ALERT_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, model_mlm, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.model_mlm = model_mlm
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score
    
    def filter_identifier(self, code, identifiers):
        code_token = get_code_tokens(code)
        filter_identifiers = []
        for identifier in identifiers:
            if is_valid_identifier(identifier):
                position = []
                for index, token in enumerate(code_token):
                    if identifier == token:
                        position.append(index)
                if not all(x > self.args.block_size - 2 for x in position):
                    filter_identifiers.append(identifier)
        return filter_identifiers


    def ga_attack(self, example, substitutes, code, initial_replace=None):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''
        code_1 = code[2]
        code_2 = code[3]

            # 先得到tgt_model针对原始Example的预测信息.

        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[6].item()
        adv_code = ''
        temp_label = None
        sucpre_code = code_1


        identifiers, code_tokens = get_identifiers(code_1, 'java')
        prog_length = len(code_tokens)
        processed_code = " ".join(code_tokens)

        identifiers_2, code_tokens_2 = get_identifiers(code_2, 'java')
        processed_code_2 = " ".join(code_tokens_2)

        
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        code_2 = " ".join(code_2.split())
        words_2 = self.tokenizer_tgt.tokenize(code_2)


        variable_names = list(substitutes.keys())
        variable_names = self.filter_identifier(code_1, variable_names)

        
        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

        nb_changed_var = 0 # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1

        # 我们可以先生成所有的substitues
        variable_substitue_dict = {}


        for tgt_word in names_positions_dict.keys():
            variable_substitue_dict[tgt_word] = substitutes[tgt_word]


        fitness_values = []
        base_chromesome = {word: word for word in variable_substitue_dict.keys()}
        population = [base_chromesome]
        # 关于chromesome的定义: {tgt_word: candidate, tgt_word_2: candidate_2, ...}
        for tgt_word in variable_substitue_dict.keys():
            # 这里进行初始化
            if initial_replace is None:
                # 对于每个variable: 选择"影响最大"的substitues
                replace_examples = []
                substitute_list = []
                
                current_prob = max(orig_prob)
                most_gap = 0.0
                initial_candidate = tgt_word
                
                # 原来是随机选择的，现在要找到改变最大的.
                for a_substitue in variable_substitue_dict[tgt_word]:
                    
                    substitute_list.append(a_substitue)
                    # 记录下这次换的是哪个substitue
                    temp_replace = get_example(code_1, tgt_word, a_substitue, "java") 
                    new_feature = convert_code_to_features(temp_replace, 
                                                            code_2,
                                                            self.tokenizer_tgt,
                                                            example[6].item(), 
                                                            self.args)
                    replace_examples.append(new_feature)

                if len(replace_examples) == 0:
                    # 并没有生成新的mutants，直接跳去下一个token
                    continue
                new_dataset = CodePairDataset(replace_examples, self.args)
                    # 3. 将他们转化成features
                logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

                _the_best_candidate = -1
                for index, temp_prob in enumerate(logits):
                    temp_label = preds[index]
                    gap = current_prob - temp_prob[temp_label]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        _the_best_candidate = index
                if _the_best_candidate == -1:
                    initial_candidate = tgt_word
                else:
                    initial_candidate = substitute_list[_the_best_candidate]
            else:
                initial_candidate = initial_replace[tgt_word]

            temp_chromesome = copy.deepcopy(base_chromesome)
            temp_chromesome[tgt_word] = initial_candidate
            population.append(temp_chromesome)
            temp_fitness, temp_label = compute_fitness(temp_chromesome, code_2, self.model_tgt, self.tokenizer_tgt, max(orig_prob), orig_label, true_label ,code_1, names_positions_dict, self.args)
            fitness_values.append(temp_fitness)

        cross_probability = 0.7

        max_iter = max(5 * len(population), 10)
        # 这里的超参数还是的调试一下.

        for i in range(max_iter):
            _temp_mutants = []
            for j in range(self.args.eval_batch_size):
                p = random.random()
                chromesome_1, index_1, chromesome_2, index_2 = select_parents(population)
                if p < cross_probability: # 进行crossover
                    if chromesome_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                        continue
                    child_1, child_2 = crossover(chromesome_1, chromesome_2)
                    if child_1 == chromesome_1 or child_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                else: # 进行mutates
                    child_1 = mutate(chromesome_1, variable_substitue_dict)
                _temp_mutants.append(child_1)
            
            # compute fitness in batch
            feature_list = []
            for mutant in _temp_mutants:
                _tmp_mutate_code = map_chromesome(mutant, code_1, "java")

                _tmp_feature = convert_code_to_features(_tmp_mutate_code, 
                                                            code_2,
                                                            self.tokenizer_tgt,
                                                            true_label,
                                                            self.args)
                feature_list.append(_tmp_feature)
            if len(feature_list) == 0:
                continue
            new_dataset = CodePairDataset(feature_list, self.args)
            mutate_logits, mutate_preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            mutate_fitness_values = []
            for index, logits in enumerate(mutate_logits):
                if mutate_preds[index] != orig_label:
                    adv_code = map_chromesome(_temp_mutants[index], code_1, "java")
                    for old_word in _temp_mutants[index].keys():
                        if old_word == _temp_mutants[index][old_word]:
                            nb_changed_var += 1
                            nb_changed_pos += len(names_positions_dict[old_word])

                    return code, prog_length, sucpre_code, adv_code, true_label, orig_label, mutate_preds[index], 1, variable_names, None, nb_changed_var, nb_changed_pos, _temp_mutants[index]
                else:
                    sucpre_code = map_chromesome(_temp_mutants[index], code_1, "java")
                _tmp_fitness = max(orig_prob) - logits[orig_label]
                mutate_fitness_values.append(_tmp_fitness)
            
            # 现在进行替换.
            for index, fitness_value in enumerate(mutate_fitness_values):
                min_value = min(fitness_values)
                if fitness_value > min_value:
                    # 替换.
                    min_index = fitness_values.index(min_value)
                    population[min_index] = _temp_mutants[index]
                    fitness_values[min_index] = fitness_value

        return code, prog_length, sucpre_code, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, nb_changed_var, nb_changed_pos, None
        
    def greedy_attack(self, example, substitutes, code):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''
            # 先得到tgt_model针对原始Example的预测信息.

        code_1 = code[2]
        code_2 = code[3]
        

        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[6].item()
        adv_code = ''
        sucpre_code = code_1
        temp_label = None


        # When do attack, we only attack the first code snippet
        identifiers, code_tokens = get_identifiers(code_1, 'java') # 只得到code_1中的identifier
        processed_code = " ".join(code_tokens)
        prog_length = len(code_tokens)

        identifiers_2, code_tokens_2 = get_identifiers(code_2, 'java')
        processed_code_2 = " ".join(code_tokens_2)

        
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)


        variable_names = list(substitutes.keys())
        variable_names = self.filter_identifier(code_1, variable_names)

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(self.args, example, 
                                                processed_code, processed_code_2,
                                                words,
                                                variable_names,
                                                self.model_tgt, 
                                                self.tokenizer_tgt, 
                                                [0,1], 
                                                batch_size=self.args.eval_batch_size, 
                                                max_length=self.args.code_length, 
                                                model_type='classification')

        if importance_score is None:
            return code, prog_length, sucpre_code, adv_code, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None


        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        # 重新计算Importance score，将所有出现的位置加起来（而不是取平均）.
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                # 这个token在code中对应的位置
                # importance_score中的位置：token_pos_to_score_pos[token_pos]
                total_score += importance_score[token_pos_to_score_pos[token_pos]]
            
            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
        # 根据importance_score进行排序

        final_code = copy.deepcopy(code_1)
        
        nb_changed_var = 0 # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}

        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]
            tgt_positions = names_positions_dict[tgt_word] # the positions of tgt_word in code
            all_substitues = substitutes[tgt_word]
            # 得到了所有位置的substitue，并使用set来去重

            most_gap = 0.0
            candidate = None
            replace_examples = []

            substitute_list = []
            # 依次记录了被加进来的substitue
            # 即，每个temp_replace对应的substitue.
            for substitute in all_substitues:
                
                substitute_list.append(substitute)
                # 记录了替换的顺序
                temp_replace = get_example(final_code, tgt_word, substitute, "java")
                # 需要将几个位置都替换成sustitue_
                new_feature = convert_code_to_features(temp_replace, 
                                                        code_2,
                                                        self.tokenizer_tgt,
                                                        example[6].item(), 
                                                        self.args)
                replace_examples.append(new_feature)
            if len(replace_examples) == 0:
                # 并没有生成新的mutants，直接跳去下一个token
                continue
            new_dataset = CodePairDataset(replace_examples, self.args)
                # 3. 将他们转化成features
            logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            assert(len(logits) == len(substitute_list))


            for index, temp_prob in enumerate(logits):
                temp_label = preds[index]
                if temp_label != orig_label:
                    # 如果label改变了，说明这个mutant攻击成功
                    is_success = 1
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate

                    adv_code = get_example(final_code, tgt_word, candidate, "java")
                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                        ('>>', tgt_word, candidate,
                        current_prob,
                        temp_prob[orig_label]), flush=True)
                    return code, prog_length, sucpre_code, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                else:
                    # 如果没有攻击成功，我们看probability的修改
                    fail_example = substitute_list[index]
                    sucpre_code = get_example(final_code, tgt_word, fail_example, 'java')
                    gap = current_prob - temp_prob[temp_label]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]
        
            if most_gap > 0:

                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = current_prob - most_gap
                final_code = get_example(final_code, tgt_word, candidate, "java")
                replaced_words[tgt_word] = candidate
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                    ('>>', tgt_word, candidate,
                    current_prob + most_gap,
                    current_prob), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word
            
            adv_code = final_code

        return code, prog_length, sucpre_code, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

class MHM_Attacker():
    def __init__(self, args, model_tgt, model_mlm, tokenizer_mlm, _token2idx, _idx2token) -> None:
        self.classifier = model_tgt
        self.model_mlm = model_mlm
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args
        self.tokenizer_mlm = tokenizer_mlm

    def filter_identifier(self, code, identifiers):
        code_token = get_code_tokens(code)
        filter_identifiers = []
        for identifier in identifiers:
            if is_valid_identifier(identifier):
                position = []
                for index, token in enumerate(code_token):
                    if identifier == token:
                        position.append(index)
                if not all(x > self.args.block_size - 2 for x in position):
                    filter_identifiers.append(identifier)
        return filter_identifiers
    
    def mcmc(self, example, substituions, tokenizer, code_pair, _label=None, _n_candi=30,
             _max_iter=100, _prob_threshold=0.95):
        code_1 = code_pair[2]
        code_2 = code_pair[3]

        # 先得到tgt_model针对原始Example的预测信息.

        logits, preds = self.classifier.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[6].item()
        adv_code = ''
        temp_label = None


        identifiers, code_tokens = get_identifiers(code_1, 'java')
        prog_length = len(code_tokens)
        processed_code = " ".join(code_tokens)

        identifiers_2, code_tokens_2 = get_identifiers(code_2, 'java')
        processed_code_2 = " ".join(code_tokens_2)

        
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)

        variable_names = list(substituions.keys())
        variable_names = self.filter_identifier(code_1, variable_names)

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return {'succ': None, 'tokens': None, 'raw_tokens': None}

        raw_tokens = copy.deepcopy(words)

        uid = get_identifier_posistions_from_code(words, variable_names)

        if len(uid) <= 0: # 是有可能存在找不到变量名的情况的.
            return {'succ': None, 'tokens': None, 'raw_tokens': None}

        # 还需要得到substitues

        variable_substitue_dict = {}
        for tgt_word in uid.keys():
            variable_substitue_dict[tgt_word] = substituions[tgt_word]
        
        old_uids = {}
        old_uid = ""
        for iteration in range(1, 1+_max_iter):
            # 这个函数需要tokens
            res = self.__replaceUID(words_2=code_2, _tokens=code_1, _label=_label, _uid=uid,
                                    substitute_dict=variable_substitue_dict,
                                    _n_candi=_n_candi,
                                    _prob_threshold=_prob_threshold)
            self.__printRes(_iter=iteration, _res=res, _prefix="  >> ")
            if res['status'].lower() in ['s', 'a']:
                if iteration == 1:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]
                flag = 0
                for k in old_uids.keys():
                    if res["old_uid"] == old_uids[k][-1]:
                        flag = 1
                        old_uids[k].append(res["new_uid"])
                        old_uid = k
                        break
                if flag == 0:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]

                code_1 = res['tokens']
                uid[res['new_uid']] = uid.pop(res['old_uid']) # 替换key，但保留value.
                variable_substitue_dict[res['new_uid']] = variable_substitue_dict.pop(res['old_uid'])
                for i in range(len(raw_tokens)):
                    if raw_tokens[i] == res['old_uid']:
                        raw_tokens[i] = res['new_uid']
                if res['status'].lower() == 's':
                    replace_info = {}
                    nb_changed_pos = 0
                    for uid_ in old_uids.keys():
                        replace_info[uid_] = old_uids[uid_][-1]
                        nb_changed_pos += len(uid[old_uids[uid_][-1]])
                    return {'succ': True, 'tokens': code_1,
                            'raw_tokens': raw_tokens, "prog_length": prog_length, "new_pred": res["new_pred"], "is_success": 1, "old_uid": old_uid, "score_info": res["old_prob"][0]-res["new_prob"][0], "nb_changed_var": len(old_uids), "nb_changed_pos":nb_changed_pos, "replace_info": replace_info, "attack_type": "MHM","orig_label": orig_label}
        replace_info = {}
        nb_changed_pos = 0
        for uid_ in old_uids.keys():
            replace_info[uid_] = old_uids[uid_][-1]
            nb_changed_pos += len(uid[old_uids[uid_][-1]])
        return {'succ': False, 'tokens': res['tokens'], 'raw_tokens': None, "prog_length": prog_length, "new_pred": res["new_pred"], "is_success": -1, "old_uid": old_uid, "score_info": res["old_prob"][0]-res["new_prob"][0], "nb_changed_var": len(old_uids), "nb_changed_pos":nb_changed_pos, "replace_info": replace_info, "attack_type": "MHM", "orig_label": orig_label}
      
    def mcmc_random(self, example, substituions, tokenizer, code_pair, _label=None, _n_candi=30,
             _max_iter=100, _prob_threshold=0.95):
        code_1 = code_pair[2]
        code_2 = code_pair[3]

        # 先得到tgt_model针对原始Example的预测信息.

        logits, preds = self.classifier.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[6].item()
        adv_code = ''
        temp_label = None


        identifiers, code_tokens = get_identifiers(code_1, 'java')
        prog_length = len(code_tokens)
        processed_code = " ".join(code_tokens)

        identifiers_2, code_tokens_2 = get_identifiers(code_2, 'java')
        processed_code_2 = " ".join(code_tokens_2)

        
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)

        variable_names = list(substituions.keys())
        variable_names = self.filter_identifier(code_1, variable_names)

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return {'succ': None, 'tokens': None, 'raw_tokens': None}

        raw_tokens = copy.deepcopy(words)

        uid = get_identifier_posistions_from_code(words, variable_names)

        if len(uid) <= 0: # 是有可能存在找不到变量名的情况的.
            return {'succ': None, 'tokens': None, 'raw_tokens': None}

        variable_substitue_dict = {}
        for tgt_word in uid.keys():
            variable_substitue_dict[tgt_word] = substituions[tgt_word]
        
        old_uids = {}
        old_uid = ""
        for iteration in range(1, 1+_max_iter):
            # 这个函数需要tokens
            res = self.__replaceUID_random(words_2=code_2, _tokens=code_1, _label=_label, _uid=uid,
                                    substitute_dict=variable_substitue_dict,
                                    _n_candi=_n_candi,
                                    _prob_threshold=_prob_threshold)
            self.__printRes(_iter=iteration, _res=res, _prefix="  >> ")
            if res['status'].lower() in ['s', 'a']:
                if iteration == 1:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]
                flag = 0
                for k in old_uids.keys():
                    if res["old_uid"] == old_uids[k][-1]:
                        flag = 1
                        old_uids[k].append(res["new_uid"])
                        old_uid = k
                        break
                if flag == 0:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]

                code_1 = res['tokens']
                uid[res['new_uid']] = uid.pop(res['old_uid']) # 替换key，但保留value.
                variable_substitue_dict[res['new_uid']] = variable_substitue_dict.pop(res['old_uid'])
                for i in range(len(raw_tokens)):
                    if raw_tokens[i] == res['old_uid']:
                        raw_tokens[i] = res['new_uid']
                if res['status'].lower() == 's':
                    replace_info = {}
                    nb_changed_pos = 0
                    for uid_ in old_uids.keys():
                        replace_info[uid_] = old_uids[uid_][-1]
                        nb_changed_pos += len(uid[old_uids[uid_][-1]])
                    return {'succ': True, 'tokens': code_1,
                            'raw_tokens': raw_tokens, "prog_length": prog_length, "new_pred": res["new_pred"], "is_success": 1, "old_uid": old_uid, "score_info": res["old_prob"][0]-res["new_prob"][0], "nb_changed_var": len(old_uids), "nb_changed_pos":nb_changed_pos, "replace_info":replace_info, "attack_type": "Ori_MHM","orig_label": orig_label}
        replace_info = {}
        nb_changed_pos = 0

        for uid_ in old_uids.keys():
            replace_info[uid_] = old_uids[uid_][-1]
            nb_changed_pos += len(uid[old_uids[uid_][-1]])
        return {'succ': False, 'tokens': res['tokens'], 'raw_tokens': None, "prog_length": prog_length, "new_pred": res["new_pred"], "is_success": -1, "old_uid": old_uid, "score_info": res["old_prob"][0]-res["new_prob"][0], "nb_changed_var": len(old_uids), "nb_changed_pos":nb_changed_pos, "replace_info": replace_info, "attack_type": "Ori_MHM", "orig_label": orig_label}

    def __replaceUID(self, words_2, _tokens, _label=None, _uid={}, substitute_dict={},
                     _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):
        
        assert _candi_mode.lower() in ["random", "nearby"]
        
        selected_uid = random.sample(substitute_dict.keys(), 1)[0] # 选择需要被替换的变量名
        if _candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_uid]
            candi_tokens = [copy.deepcopy(_tokens)]
            candi_labels = [_label]
            for c in random.sample(substitute_dict[selected_uid], min(_n_candi, len(substitute_dict[selected_uid]))): # 选出_n_candi数量的候选.
                if isUID(c): # 判断是否是变量名.
                    candi_token.append(c)
                    candi_tokens.append(copy.deepcopy(_tokens))
                    candi_labels.append(_label)
                    candi_tokens[-1] = get_example(candi_tokens[-1], selected_uid, c, "java")

            new_example = []
            for tmp_tokens in candi_tokens:
                new_feature = convert_code_to_features(tmp_tokens, 
                                                words_2,
                                                self.tokenizer_mlm,
                                                _label,
                                                self.args)
                new_example.append(new_feature)
            new_dataset = CodePairDataset(new_example, self.args)
            prob, pred = self.classifier.get_results(new_dataset, self.args.eval_batch_size)

            for i in range(len(candi_token)):   # Find a valid example
                if pred[i] != _label: # 如果有样本攻击成功
                    return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                            "old_uid": selected_uid, "new_uid": candi_token[i],
                            "old_prob": prob[0], "new_prob": prob[i],
                            "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}

            candi_idx = 0
            min_prob = 1.0

            for idx, a_prob in enumerate(prob[1:]):
                if a_prob[_label] < min_prob:
                    candi_idx = idx + 1
                    min_prob = a_prob[_label]

            # 找到Ground_truth对应的probability最小的那个mutant
            # At last, compute acceptance rate.
            alpha = (1-prob[candi_idx][_label]+1e-10) / (1-prob[0][_label]+1e-10)
            # 计算这个id对应的alpha值.
            if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
                return {"status": "r", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
            else:
                return {"status": "a", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
        else:
            pass

    def __replaceUID_random(self, words_2, _tokens, _label=None, _uid={}, substitute_dict={},
                     _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):
        
        assert _candi_mode.lower() in ["random", "nearby"]
        
        selected_uid = random.sample(substitute_dict.keys(), 1)[0] # 选择需要被替换的变量名
        if _candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_uid]
            candi_tokens = [copy.deepcopy(_tokens)]
            candi_labels = [_label]
            for c in random.sample(self.idx2token, _n_candi): # 选出_n_candi数量的候选.
                if c in _uid.keys():
                    continue
                if isUID(c): # 判断是否是变量名.
                    candi_token.append(c)
                    candi_tokens.append(copy.deepcopy(_tokens))
                    candi_labels.append(_label)
                    candi_tokens[-1] = get_example(candi_tokens[-1], selected_uid, c, "java")

            new_example = []
            for tmp_tokens in candi_tokens:
                new_feature = convert_code_to_features(tmp_tokens, 
                                                words_2,
                                                self.tokenizer_mlm,
                                                _label,
                                                self.args)
                new_example.append(new_feature)
            new_dataset = CodePairDataset(new_example, self.args)
            prob, pred = self.classifier.get_results(new_dataset, self.args.eval_batch_size)

            for i in range(len(candi_token)):   # Find a valid example
                if pred[i] != _label: # 如果有样本攻击成功
                    return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                            "old_uid": selected_uid, "new_uid": candi_token[i],
                            "old_prob": prob[0], "new_prob": prob[i],
                            "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}

            candi_idx = 0
            min_prob = 1.0

            for idx, a_prob in enumerate(prob[1:]):
                if a_prob[_label] < min_prob:
                    candi_idx = idx + 1
                    min_prob = a_prob[_label]

            # 找到Ground_truth对应的probability最小的那个mutant
            # At last, compute acceptance rate.
            alpha = (1-prob[candi_idx][_label]+1e-10) / (1-prob[0][_label]+1e-10)
            # 计算这个id对应的alpha值.
            if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
                return {"status": "r", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
            else:
                return {"status": "a", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
        else:
            pass

    def __printRes(self, _iter=None, _res=None, _prefix="  => "):
        if _res['status'].lower() == 's':   # Accepted & successful
            print("%s iter %d, SUCC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'r': # Rejected
            print("%s iter %d, REJ. %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'a': # Accepted
            print("%s iter %d, ACC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
            
class WIR_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, _token2idx, _idx2token) -> None:
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args

    def filter_identifier(self, code, identifiers):
        code_token = get_code_tokens(code)
        filter_identifiers = []
        for identifier in identifiers:
            if is_valid_identifier(identifier):
                position = []
                for index, token in enumerate(code_token):
                    if identifier == token:
                        position.append(index)
                if not all(x > self.args.block_size - 2 for x in position):
                    filter_identifiers.append(identifier)
        return filter_identifiers

    def wir_random_attack(self, example, code):
        code_1 = code[2]
        code_2 = code[3]

        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[6].item()
        adv_code = ''
        temp_label = None

        # When do attack, we only attack the first code snippet
        identifiers, code_tokens = get_identifiers(code_1, 'java')  # 只得到code_1中的identifier
        processed_code = " ".join(code_tokens)
        prog_length = len(code_tokens)

        identifiers_2, code_tokens_2 = get_identifiers(code_2, 'java')
        processed_code_2 = " ".join(code_tokens_2)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_tgt)
        # code_2 = " ".join(code_2.split())
        # words_2 = self.tokenizer_tgt.tokenize(code_2)

        variable_names = [identifier[0] for identifier in identifiers]
        variable_names = self.filter_identifier(code_1, variable_names)

        if not orig_label == true_label:
            print("skip for wrong predict")
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(self.args, example,
                                                                                               processed_code,
                                                                                               processed_code_2,
                                                                                               words,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt,
                                                                                               [0, 1],
                                                                                               batch_size=self.args.eval_batch_size,
                                                                                               max_length=self.args.block_size,
                                                                                               model_type='classification')

        if importance_score is None:
            return code, prog_length, adv_code, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None

        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        # 重新计算Importance score，将所有出现的位置加起来（而不是取平均）.
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                # 这个token在code中对应的位置
                # importance_score中的位置：token_pos_to_score_pos[token_pos]
                total_score += importance_score[token_pos_to_score_pos[token_pos]]

            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
        # 根据importance_score进行排序

        final_code = copy.deepcopy(code_1)

        nb_changed_var = 0  # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}

        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]
            tgt_positions = names_positions_dict[tgt_word]  # the positions of tgt_word in code

            all_substitues = []
            num = 0
            while num < 30:
                tmp_var = random.choice(self.idx2token)
                if isUID(tmp_var):
                    all_substitues.append(tmp_var)
                    num += 1

            most_gap = 0.0
            candidate = None
            replace_examples = []

            substitute_list = []
            # 依次记录了被加进来的substitue
            # 即，每个temp_replace对应的substitue.
            for substitute in all_substitues:
                substitute_list.append(substitute)
                # 记录了替换的顺序
                temp_replace = get_example(final_code, tgt_word, substitute, 'java')
                # 需要将几个位置都替换成sustitue_
                new_feature = convert_code_to_features(temp_replace,code_2,self.tokenizer_tgt,example[6].item(), self.args)
                replace_examples.append(new_feature)
            if len(replace_examples) == 0:
                # 并没有生成新的mutants，直接跳去下一个token
                continue
            new_dataset = CodePairDataset(replace_examples, self.args)
            # 3. 将他们转化成features
            logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            assert (len(logits) == len(substitute_list))

            for index, temp_prob in enumerate(logits):
                temp_label = preds[index]
                if temp_label != orig_label:
                    # 如果label改变了，说明这个mutant攻击成功
                    is_success = 1
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate

                    adv_code = get_example(final_code, tgt_word, candidate, 'java')
                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                          ('>>', tgt_word, candidate,
                           current_prob,
                           temp_prob[orig_label]), flush=True)
                    return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                else:
                    # 如果没有攻击成功，我们看probability的修改
                    gap = current_prob - temp_prob[temp_label]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]

            if most_gap > 0:

                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = current_prob - most_gap
                final_code = get_example(final_code, tgt_word, candidate, 'java')
                replaced_words[tgt_word] = candidate
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                      ('>>', tgt_word, candidate,
                       current_prob + most_gap,
                       current_prob), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word

            adv_code = final_code

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

class RNNS_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, model_mlm, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.model_mlm = model_mlm
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score
        self.variable_emb, self.variable_name = self._get_variable_info()
    
    def _filter_identifier(self, code, identifiers):
        code_token = get_code_tokens(code)
        filter_identifiers = []
        for identifier in identifiers:
            if is_valid_identifier(identifier):
                position = []
                for index, token in enumerate(code_token):
                    if identifier == token:
                        position.append(index)
                if not all(x > self.args.block_size - 2 for x in position):
                    filter_identifiers.append(identifier)
        return filter_identifiers

    def _get_variable_info(self):
        codes = []
        variables = []
        variable_embs = []

        cache_path = "../dataset/rnns_cache.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:        
            codebert_mlm = RobertaForMaskedLM.from_pretrained(self.args.base_model)
            tokenizer_mlm = RobertaTokenizer.from_pretrained(self.args.base_model)
            codebert_mlm.to('cuda')
            
            for path in [self.args.train_data_file, self.args.valid_data_file, self.args.test_data_file]:
                url_to_code = {}
                with open('/'.join(path.split('/')[:-1]) + '/data.jsonl') as f:
                    for line in f:
                        line = line.strip()
                        js = json.loads(line)
                        url_to_code[js['idx']] = js['func']
                
                with open(path) as rf:
                    for line in rf:
                        line = line.strip()
                        url1, url2, label = line.split(' ')
                        url1 = int(url1)
                        url2 = int(url2)
                        if url1 not in url_to_code or url2 not in url_to_code:
                            continue
                        code = url_to_code[url1]
                        codes.append(code)

            for code in codes:
                try:
                    identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(code, "java"), "java")
                except:
                    identifiers, code_tokens = get_identifiers(code, "java")

                cur_variables = []
                for name in identifiers:
                    if ' ' in name[0].strip() and not is_valid_variable_name(name[0], lang='java'):
                        continue
                    cur_variables.append(name[0])

                variables.extend(cur_variables)

            variables = list(set(variables))
            for var in tqdm(variables):
                sub_words = tokenizer_mlm.tokenize(var)
                sub_words = [tokenizer_mlm.cls_token] + sub_words[:self.args.block_size - 2] + [tokenizer_mlm.sep_token]
                input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
                with torch.no_grad():
                    orig_embeddings = codebert_mlm.roberta(input_ids_.to('cuda'))[0][0][1:-1]
                    mean_embedding = torch.mean(orig_embeddings, dim=0, keepdim=True).cpu().detach().numpy()[0]
                    variable_embs.append(mean_embedding)

            assert len(variable_embs) == len(variables)

            with open(cache_path, 'wb') as f:
                pickle.dump((variable_embs, variables), f)

            return variable_embs, variables
    
    def _get_var_importance_score_by_uncertainty(self, args, example, code, code_2, words_list: list, variable_names: list,
                                             tgt_model,
                                             tokenizer, batch_size=16, max_length=512,
                                             model_type='classification'):


        positions = get_identifier_posistions_from_code(words_list, variable_names)

        if len(positions) == 0:
            return None

        new_example = []

        # code2_tokens, _, _ = _tokenize(code_2, tokenizer)
        masked_token_list, masked_var_list = get_replaced_var_code_with_meaningless_char(words_list, positions)
        # replace_token_positions 表示着，哪一个位置的token被替换了.

        for index, tokens in enumerate([words_list] + masked_token_list):
            new_code = ' '.join(tokens)
            new_feature = convert_code_to_features(new_code, code_2, tokenizer, example[6].item(), args)
            # new_feature = convert_examples_to_features(tokens, code2_tokens, example[1].item(), None, None, tokenizer,
            #                                        args, None)
            new_example.append(new_feature)
        new_dataset = CodePairDataset(new_example, self.args)

        logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
        orig_probs = logits[0]
        orig_label = preds[0]

        var_pos_delt_prob_disp = {}
        var_neg_delt_prob_disp = {}
        var_importance_score_by_variance = {}
        for prob, var in zip(logits[1:], masked_var_list):
            if var in var_pos_delt_prob_disp:
                var_pos_delt_prob_disp[var].append(prob[orig_label])
                var_neg_delt_prob_disp[var].append(1 - prob[orig_label])
            else:
                var_pos_delt_prob_disp[var] = [prob[orig_label]]
                var_neg_delt_prob_disp[var] = [1 - prob[orig_label]]

        for var in var_pos_delt_prob_disp:
            MaxP = np.max(var_pos_delt_prob_disp[var] + var_neg_delt_prob_disp[var])
            VarP = (np.var(var_pos_delt_prob_disp[var]) + np.var(var_neg_delt_prob_disp[var])) / 2

            # var_importance_score_by_variance[var] = VarP/MaxP
            var_importance_score_by_variance[var] = VarP

        return var_importance_score_by_variance, positions


    def attack(self, example, code):
        code_1 = code[2]
        code_2 = code[3]

        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[6].item()
        adv_code = ''
        temp_label = None
        true_label_prob = orig_prob[true_label]

        try:
            identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(code_1, "java"), "java")
        except:
            identifiers, code_tokens = get_identifiers(code_1, "java")

        try:
            identifiers_2, code_tokens_2 = get_identifiers(remove_comments_and_docstrings(code_2, "java"), "java")
        except:
            identifiers_2, code_tokens_2 = get_identifiers(code_2, "java")
        
        variable_names = []
        for name in identifiers:
            if ' ' in name[0].strip() and not is_valid_variable_name(name[0], lang='java'):
                continue
            variable_names.append(name[0])
        
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)
        processed_code_2 = " ".join(code_tokens_2)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)

        substituions = {}

        if not orig_label == true_label:
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        if len(variable_names) == 0:
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        names_to_importance_score, names_positions_dict = self._get_var_importance_score_by_uncertainty(self.args, example,
                                                                                               processed_code,
                                                                                               processed_code_2,
                                                                                               words,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt,
                                                                                               batch_size=self.args.eval_batch_size,
                                                                                               max_length=self.args.block_size,
                                                                                               model_type='classification')
        

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)

        ranking_order_variance = 0
        var_size = len(names_to_importance_score.keys())
        

        final_words = copy.deepcopy(words)
        final_code = copy.deepcopy(code_1)
        nb_changed_var = 0  
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}

        for name_and_score in sorted_list_of_names:
            used_candidate = list(replaced_words.values())
            tgt_word = name_and_score[0]
            tgt_word_len = len(tgt_word)
            tgt_positions = names_positions_dict[tgt_word]

            tgt_index = self.variable_name.index(tgt_word)
            distances = 1 - cosine_similarity(np.array(self.variable_emb),np.array([self.variable_emb[tgt_index]]))
            variable_index_list = [i for i,  distance in enumerate(distances) if distance < self.args.max_distance and len(self.variable_name[i])<= tgt_word_len + self.args.max_length_diff]
            variable_embs = [ self.variable_emb[index] for index in variable_index_list]
            valid_variable_names = [ self.variable_name[index] for index in variable_index_list]
            

            index_list = [i for i in range(0, len(valid_variable_names))]
            random.shuffle(index_list)
            inds_1 = index_list[:self.args.substitutes_size]
            inds_2 = index_list[self.args.substitutes_size:3*self.args.substitutes_size]
            all_substitues = [valid_variable_names[ind] for ind in inds_1]
            substituions[tgt_word] =  [valid_variable_names[ind] for ind in inds_2]

 

            candidate = None
            new_substitutes = []
            for sub in all_substitues:
                if sub not in used_candidate:
                    new_substitutes.append(sub)
                       
            best_candidate = tgt_word
            loop_time = 0
            momentum = None
            track = []
            while True:
                substitute_list = []
                replace_examples = []
                most_gap = 0.0
                for substitute in new_substitutes:
                    substitute_list.append(substitute)
                    temp_code = get_example(final_code, tgt_word, substitute, "java")
                    new_feature = convert_code_to_features(temp_code, code_2, self.tokenizer_tgt, example[6].item(), self.args)
                    replace_examples.append(new_feature)
                if len(replace_examples) == 0:
                    break

                new_dataset = CodePairDataset(replace_examples, self.args)
    
                logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
                assert (len(logits) == len(substitute_list))
                used_candidate.extend(all_substitues)
                
                golden_prob_decrease_track = {}
                for index, temp_prob in enumerate(logits):
                    temp_label = preds[index]
                    if temp_label != orig_label:
                        is_success = 1
                        nb_changed_var += 1
                        nb_changed_pos += len(names_positions_dict[tgt_word])
                        candidate = substitute_list[index]
                        replaced_words[tgt_word] = candidate
                        adv_code = get_example(final_code, tgt_word, candidate, "java")
                        print("%s SUC! %s => %s (%.5f => %.5f)" % \
                              ('>>', tgt_word, candidate,
                               current_prob,
                               temp_prob[orig_label]), flush=True)
                        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                    else:
                        gap = current_prob - temp_prob[temp_label]
                        if gap > 0:
                            golden_prob_decrease_track[substitute_list[index]] = gap

                if len(golden_prob_decrease_track) > 0:
                    cur_iter_track = {}
                    sorted_golden_prob_decrease_track = sorted(golden_prob_decrease_track.items(), key=lambda x: x[1],
                                                               reverse=True)
                    (candidate, most_gap) = sorted_golden_prob_decrease_track[0]
                    cur_iter_track["candidate"] = list(map(lambda x: x[0], sorted_golden_prob_decrease_track[1:]))

                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    current_prob = current_prob - most_gap
                    if candidate not in valid_variable_names or best_candidate not in valid_variable_names:
                        replaced_words[tgt_word] = best_candidate
                        final_code = get_example(final_code, tgt_word, best_candidate, "java")
                        adv_code = final_code
                        break

                    candidate_index = valid_variable_names.index(candidate)
                    best_candidate_index = valid_variable_names.index(best_candidate)

                        
                    prob_delt_emb = variable_embs[candidate_index] - variable_embs[best_candidate_index]
                    if momentum is None:
                        momentum = prob_delt_emb   
                    momentum = (1 - self.args.a) * momentum + self.args.a * prob_delt_emb
                    
                    if self.args.rnns_type == "RNNS-Delta":
                        virtual_emb = variable_embs[candidate_index] + prob_delt_emb
                    elif self.args.rnns_type == "RNNS-Smooth":
                        virtual_emb = variable_embs[candidate_index] + momentum
                    elif self.args.rnns_type == "RNNS-Raw":
                        virtual_emb = variable_embs[candidate_index]
                    else:
                        pass
                    
                    similarity = cosine_similarity(np.array(variable_embs),
                                                   np.array([virtual_emb]))


                    inds = heapq.nlargest(len(similarity), range(len(similarity)), similarity.__getitem__)
                    new_substitutes.clear()
                    if len(replaced_words) > 0:
                        used_candidate.extend(list(replaced_words.values()))

                    for ind in inds:
                        temp_var = valid_variable_names[ind]
                        if temp_var not in used_candidate:
                            new_substitutes.append(temp_var)
                            if temp_var in substituions[tgt_word]:
                                substituions[tgt_word].remove(temp_var)
                            used_candidate.append(temp_var)
                            if len(new_substitutes) >= self.args.substitutes_size:
                                break

                    best_candidate = candidate
                    cur_iter_track["best_candidate"] = best_candidate
                    track.append(cur_iter_track)

                else:
                    if best_candidate != tgt_word:
                        replaced_words[tgt_word] = best_candidate
                        final_code = get_example(final_code, tgt_word, best_candidate, "java")
                        adv_code = final_code
                        print("%s ACC! %s => %s (%.5f => %.5f)" % \
                              ('>>', tgt_word, best_candidate,
                               current_prob + most_gap,
                               current_prob), flush=True)

                    break

                loop_time += 1
                if loop_time >= self.args.iters:
                    if best_candidate != tgt_word:
                        replaced_words[tgt_word] = best_candidate
                        final_code = get_example(final_code, tgt_word, best_candidate, "java")
                        adv_code = final_code
                        print("%s ACC! %s => %s (%.5f => %.5f)" % \
                              ('>>', tgt_word, best_candidate,
                               current_prob + most_gap,
                               current_prob), flush=True)
                    break

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
 
class Beam_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, model_mlm):
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm
        self.model_mlm = model_mlm

    def is_vaild(self, code_token, identifier):
        if not is_valid_identifier(identifier):
            return False
        position = []
        for index, token in enumerate(code_token):
            if identifier == token:
                position.append(index)
        if all(x > self.args.block_size-2 for x in position):
            return False
        return True

    def perturb(self, example, code_1, code_2, all_substitues, tgt_word, iters, equal=False):
        is_success = -1
        final_code = copy.deepcopy(code_1)
        tmp_code_2 = copy.deepcopy(code_2)
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        # print("attacker true label: ", orig_label)
        current_prob = max(orig_prob)

        candidate = None
        substitute_list = []
        all_substitues = list(set([subs.strip() for subs in all_substitues if subs != tgt_word]))
        cosine_list = []
        for sub in all_substitues:
            temp_code = get_example(code_1, tgt_word, sub, 'java')
            code1_tokens = [self.tokenizer_mlm.cls_token] + self.tokenizer_mlm.tokenize(code_1)[
                                                            :self.args.block_size - 2] + [self.tokenizer_mlm.sep_token]
            code2_tokens = [self.tokenizer_mlm.cls_token] + self.tokenizer_mlm.tokenize(temp_code)[
                                                            :self.args.block_size - 2] + [self.tokenizer_mlm.sep_token]
            code1_ids = self.tokenizer_mlm.convert_tokens_to_ids(code1_tokens)
            code2_ids = self.tokenizer_mlm.convert_tokens_to_ids(code2_tokens)
            context_embeddings1 = self.model_mlm(torch.tensor(code1_ids)[None, :].to(self.args.device))[0]
            context_embeddings1 = context_embeddings1.reshape(context_embeddings1.size()[0],
                                                              context_embeddings1.size()[1] *
                                                              context_embeddings1.size()[2])
            context_embeddings2 = self.model_mlm(torch.tensor(code2_ids)[None, :].to(self.args.device))[0]
            context_embeddings2 = context_embeddings2.reshape(context_embeddings2.size()[0],
                                                              context_embeddings2.size()[1] *
                                                              context_embeddings2.size()[2])
            try:
                cosine_similarity = torch.cosine_similarity(context_embeddings1, context_embeddings2, dim=1).item()
                cosine_list.append(cosine_similarity)
            except:
                cosine_list.append(0)
        subs_dict = dict(zip(all_substitues, cosine_list))
        subs_dict = dict(sorted(subs_dict.items(), key=lambda x: x[1], reverse=True))
        select_substitues = list(subs_dict.keys())[:30]
        gaps = []
        replace_examples = []

        for substitute in select_substitues:
            if not is_valid_identifier(substitute.strip()):
                continue
            substitute_list.append(substitute.strip())
            temp_replace = get_example(final_code, tgt_word, substitute.strip(), 'java')
            new_feature = convert_code_to_features(temp_replace, 
                                                    code_2,
                                                    self.tokenizer_tgt,
                                                    example[6].item(), 
                                                    self.args)
            replace_examples.append(new_feature)
        new_dataset = CodePairDataset(replace_examples, self.args)
        logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
        for index, temp_prob in enumerate(logits):
            candidate = substitute_list[index]
            temp_label = preds[index]
            if temp_label != orig_label:
                # print("attacker predict label: ", temp_label)
                is_success = 1
                adv_code = get_example(final_code, tgt_word, candidate, 'java')
                return [[is_success, adv_code, candidate, temp_prob[orig_label]]]
            elif equal is True and temp_prob[temp_label] <= current_prob:
                gaps.append(
                    [is_success, get_example(final_code, tgt_word, candidate, 'java'), candidate, temp_prob[temp_label]])
            elif equal is False and temp_prob[temp_label] < current_prob:
                gaps.append(
                    [is_success, get_example(final_code, tgt_word, candidate, 'java'), candidate, temp_prob[temp_label]])

        if len(gaps) > 0:
            return gaps
        else:
            return []

    def beam_attack(self, orig_prob, example, substitutes, code_pair, statement_dict, beam_size):
        state_weight = {"For": 26.05, "Try": 23.89, "If": 23.37, "Method": 17.9, "Throw": 13.39, "Return": 11.26}
        first_probability = 26.05
        state_list = list(statement_dict.keys())
        label = example[6].item()
        result = {"succ": -1}
        code_1 = code_pair[2]
        code_2 = code_pair[3]
        code_2 = " ".join(code_2.split())
        words_2 = self.tokenizer_tgt.tokenize(code_2)

        # start beam attack
        iter = 0
        init_pop = {}
        final_pop = {}
        used_iden = []
        replace_info = ""
        tmp_code = code_1
        code_token = get_code_tokens(tmp_code)
        for key, identifiers in statement_dict.items():
            if iter == 0:
                used_iden += identifiers
                for identifier in identifiers:
                    if not self.is_vaild(code_token, identifier):
                        continue
                    gaps = self.perturb(example, tmp_code, code_2, substitutes[identifier], identifier, iter)
                    if len(gaps) > 0:
                        for gap in gaps:
                            is_success, final_code, candidate, current_prob = gap[0], gap[1], gap[2], gap[3]
                            if candidate is not None:
                                sequence = [iden for iden in identifiers if iden != identifier]
                                replace_info = identifier + ':' + candidate + ','
                                init_pop[replace_info] = {"adv_code": final_code, "prob": current_prob, "original_var": [identifier],
                                                        "adv_var": [candidate], "sequence": sequence}
                                if is_success == 1:
                                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                                          ('>>', identifier, candidate,
                                           orig_prob,
                                           current_prob), flush=True)
                                    result["succ"] = 1
                                    result["adv_code"] = final_code
                                    result["replace_info"] = replace_info
                                    result["type"] = "Beam"
                                    return result
                    else:
                        init_pop["noChange"] = {"adv_code": code_1, "prob": orig_prob, "original_var": [],
                                                "adv_var": [], "sequence": identifiers}

                final_pop = dict(sorted(init_pop.items(), key=lambda x: x[1]['prob'])[:beam_size])

            num_iter = len(identifiers) - 1
            if iter > 0:
                tmp_pop = {}
                identifiers = [iden for iden in identifiers if iden not in used_iden]
                used_iden += identifiers
                final_pop_copy = copy.copy(final_pop)
                for replace_info, value in final_pop_copy.items():
                    tmp_pop[replace_info] = {"adv_code": value["adv_code"], "prob": value["prob"], "original_var": value["original_var"],
                                                   "adv_var": value["adv_var"], "sequence": identifiers}
                final_pop = tmp_pop
                state = state_list[iter]
                if state in state_weight:
                    probability = state_weight[state]
                    num_iter = math.ceil(len(identifiers) * probability / first_probability)
                else:
                    probability = state_weight.get(list(state_weight.keys())[-1])
                    num_iter = math.ceil(len(identifiers) * probability / first_probability)

            for i_iter in range(num_iter):
                tmp_pop = {}
                final_pop_copy = copy.copy(final_pop)
                for replace_info, value in final_pop_copy.items():
                    if len(value["sequence"]) == 0:
                        continue
                    for seq in value["sequence"]:
                        if not self.is_vaild(code_token, seq):
                            continue
                        new_feature = convert_code_to_features(value["adv_code"], code_2, self.tokenizer_tgt, example[6].item(), self.args)
                        new_example = CodePairDataset([new_feature], self.args)
                        gaps = self.perturb(new_example[0], value["adv_code"], code_2, substitutes[seq], seq, iter)
                        if len(gaps) > 0:
                            for gap in gaps:
                                is_success, final_code, candidate, current_prob = gap[0], gap[1], gap[2], gap[3]
                                if candidate is not None:
                                    original_var = value["original_var"] + [seq]
                                    adv_var = value["adv_var"] + [candidate]
                                    new_replace_info = ''
                                    for info_i in range(len(original_var)):
                                        new_replace_info += original_var[info_i] + ':' + adv_var[info_i] + ','
                                    sequence = [iden for iden in value["sequence"] if iden not in original_var]
                                    tmp_pop[new_replace_info] = {"adv_code": final_code, "prob": current_prob, "original_var": original_var,
                                                        "adv_var": adv_var, "sequence": sequence}
                                    if is_success == 1:
                                        print("%s SUC! %s => %s (%.5f => %.5f)" % \
                                              ('>>', original_var, adv_var,
                                               orig_prob,
                                               current_prob), flush=True)
                                        result["succ"] = 1
                                        result["adv_code"] = final_code
                                        result["replace_info"] = new_replace_info
                                        result["type"] = "Beam"
                                        return result
                        else:
                            tmp_pop[replace_info] = value


                select_dict = dict(list(tmp_pop.items()) + list(final_pop_copy.items()))
                final_pop = dict(sorted(select_dict.items(), key=lambda x: x[1]['prob'])[:beam_size])
                if operator.eq(list(final_pop.keys()), list(final_pop_copy.keys())):
                    break
                if i_iter != num_iter:
                    duplicate_key = [i for i in list(final_pop.keys()) if i in list(final_pop_copy.keys())]
                    if len(duplicate_key) > 0:
                        for pop_key in duplicate_key:
                            del final_pop[pop_key]

            final_pop = final_pop
            iter += 1

        # final iter
        max_len = 0
        for replace_info, value in final_pop.items():
            if len(value["original_var"]) > max_len:
                final_pop = {replace_info: value}
                max_len = len(value["original_var"])

        replace_identifier = []
        adv_identifier = []
        for replace_info, value in final_pop.items():
            replace_identifier += value["original_var"]
            adv_identifier += value["adv_var"]
        replace_dict = {}
        for identifier, adv in zip(replace_identifier, adv_identifier):
            if adv not in list(replace_dict.keys()):
                subs = substitutes[identifier]
                subs = list(set([sub.strip() for sub in subs]))
                subs.remove(adv)
                subs.append(identifier)
                replace_dict[adv] = subs
        new_pop = {}
        for replace_info, value in final_pop.items():
            new_pop[replace_info] = {"adv_code": value["adv_code"], "prob": value["prob"],
                                     "original_var": value["original_var"],
                                     "adv_var": value["adv_var"], "sequence": value["adv_var"]}
        flag = 0
        for i_iter in range(len(adv_identifier)):
            if i_iter > 0 and flag == 0:
                break
            tmp_pop = {}
            final_pop_copy = copy.copy(new_pop)
            for replace_info, value in final_pop_copy.items():
                if len(value["sequence"]) == 0:
                    continue
                for seq in value["sequence"]:
                    try:
                        code_token = get_code_tokens(value["adv_code"])
                    except:
                        print("syntax errors!")
                        continue
                    if not self.is_vaild(code_token, seq):
                        continue
                    new_feature = convert_code_to_features(value["adv_code"], code_2, self.tokenizer_tgt, example[6].item(), self.args)
                    new_example = CodePairDataset([new_feature], self.args)
                    gaps = self.perturb(new_example[0], value["adv_code"], code_2, replace_dict[seq], seq, iter, equal=True)
                    flag += len(gaps)
                    if len(gaps) > 0:
                        gap = gaps[0]
                        is_success, final_code, candidate, current_prob = gap[0], gap[1], gap[2], gap[3]
                        if candidate is not None:
                            original_var, adv_var = [], []
                            if candidate in value["original_var"]:
                                value["original_var"].remove(candidate)
                                value["adv_var"].remove(seq)
                                original_var = value["original_var"]
                                adv_var = [candidate if i == seq else i for i in value["adv_var"]]
                            else:
                                original_var = value["original_var"]
                                adv_var = [candidate if i == seq else i for i in value["adv_var"]]
                            new_replace_info = ''
                            for info_i in range(len(original_var)):
                                new_replace_info += original_var[info_i] + ':' + adv_var[info_i] + ','
                            sequence = [iden for iden in value["sequence"] if iden not in adv_var]
                            tmp_pop[new_replace_info] = {"adv_code": final_code, "prob": current_prob,
                                                         "original_var": original_var,
                                                         "adv_var": adv_var, "sequence": sequence}
                            if is_success == 1:
                                print("%s SUC in Final! %s => %s (%.5f => %.5f)" % \
                                      ('>>', original_var, adv_var,
                                       orig_prob,
                                       0.0), flush=True)
                                result["succ"] = 1
                                result["adv_code"] = final_code
                                result["replace_info"] = new_replace_info
                                result["type"] = "Beam"
                                return result
                    else:
                        tmp_pop[replace_info] = value
            select_dict = dict(list(tmp_pop.items()) + list(final_pop_copy.items()))
            new_pop = dict(sorted(select_dict.items(), key=lambda x: x[1]['prob'])[:beam_size])
            if operator.eq(list(new_pop.keys()), list(final_pop_copy.keys())):
                break

        return result
