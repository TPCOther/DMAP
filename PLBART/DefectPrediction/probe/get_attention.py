import sys
sys.path.append('../code')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import pickle
import argparse
from tqdm import tqdm
from model_encoder import Model
from run import TextDataset
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer, PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer
from datasets import load_dataset
from torch.nn import functional as F
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--block_size", default=510, type=int)

parser.add_argument("--model_name", default="codet5", type=str,
                        help="The name of model which will be attacked.")

args =parser.parse_known_args()[0]
args.device = torch.device("cuda")

# target = "attack_cross"

dataset = load_dataset(
    'json',
    data_files={
        'test': f"./train_samples_encoder.jsonl",
    })
config = PLBartConfig.from_pretrained('uclanlp/plbart-base', num_labels = 4)
tokenizer = AutoTokenizer.from_pretrained('uclanlp/plbart-base')
model = PLBartForConditionalGeneration.from_pretrained('uclanlp/plbart-base', config=config)

def get_ids(code):
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return torch.tensor(source_ids)

def convert_examples_to_features(js):
    original_code = get_ids(js['code'])
    return {
        "original_code": original_code,
        "label": js['orig_label']
    }

dataset = dataset.map(convert_examples_to_features)

model = Model(model, config, tokenizer, args)
model.load_state_dict(torch.load("../saved_models/checkpoint-best-f1/codet5_model.bin"), strict=False)
model.to(args.device)
model.eval()

eval_dataset = dataset["test"]

for m in range(1,6):
    train_data = []
    ffns_data = []
    batch_size = 16
    for k in tqdm(range(0, len(eval_dataset), batch_size)):
        example = eval_dataset[k:k+batch_size]
        atts, _, _, _ = model.get_attentions_output(example["original_code"], m, 16)
        atts = atts[:, 0, :].detach().cpu()

        for i in range(len(example["original_code"])):
            train_data.append((atts[i], example["orig_label"][i]))

    torch.save(train_data, f"./att/train_attentions_{m}th.pt")
