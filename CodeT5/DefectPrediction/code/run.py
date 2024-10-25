from __future__ import absolute_import, division, print_function

import logging
import os
import pickle
import sys
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../python_parser')


import torch
from torch.utils.data import Dataset
import json
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                        T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          )

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 }


class InputFeatures(object):
    def __init__(self, input_tokens, input_ids, idx, label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label

def convert_examples_to_features(code, label, tokenizer, args):
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens, source_ids, 0, label)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        file_type = file_path.split('/')[-1].split('.')[0]
        folder = '/'.join(file_path.split('/')[:-1])

        cache_file_path = os.path.join(folder, '{}_cached_{}'.format(args.model_name, file_type))
        code_pairs_file_path = os.path.join(folder, '{}_cached_{}.pkl'.format(args.model_name, file_type))

        print('\n cached_features_file: ', cache_file_path)
        try:
            self.examples = torch.load(cache_file_path)
            with open(code_pairs_file_path, 'rb') as f:
                code_files = pickle.load(f)
        except:
            code_files = []
            with open(file_path) as f:
                for line in f:
                    js = json.loads(line.strip())
                    code = js['func']
                    code = code.replace("\\n", "\n").replace('\"', '"')
                    label = js['target']
                    self.examples.append(convert_examples_to_features(code, int(label), tokenizer,args))
                    code_files.append(code)
            assert(len(self.examples) == len(code_files))
            with open(code_pairs_file_path, 'wb') as f:
                pickle.dump(code_files, f)
            torch.save(self.examples, cache_file_path)
        
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)

