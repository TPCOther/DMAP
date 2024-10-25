import sys
sys.path.append('../code')
sys.path.append('../attack')
sys.path.append('../../../python_parser')
from model import Model
from run import TextDataset
from tqdm import tqdm
import json
import argparse
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch
from datasets import load_dataset

parser = argparse.ArgumentParser()

parser.add_argument("--block_size", default=510, type=int)

parser.add_argument("--model_name", default="codebert", type=str,
                        help="The name of model which will be attacked.")

args =parser.parse_known_args()[0]
args.device = torch.device("cuda")

config = RobertaConfig.from_pretrained('microsoft/codebert-base')
config.num_labels = 4
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
source_model = RobertaModel.from_pretrained('microsoft/codebert-base', config=config)

model = Model(source_model, config, tokenizer, args)

model.load_state_dict(torch.load("../saved_models/checkpoint-best-f1/codebert_model.bin"), strict=False)
model.to(args.device)
model.eval()

def get_ids(code):
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return torch.tensor(source_ids)

eval_dataset = TextDataset(tokenizer, args, "../dataset/train.jsonl")

# Load original source codes
source_codes = []
labels = []
with open("../dataset/train.jsonl") as f:
    for line in f:
        js=json.loads(line.strip())
        code = js['func']
        source_codes.append(code)
assert(len(source_codes) == len(eval_dataset))

batch_size = 16
with open("./train_samples.jsonl", "w") as wf:
        batch = []
        i = 0
        for index, example in enumerate(tqdm(eval_dataset)):
            if i < batch_size:
                batch.append(example)
                i += 1
            if i == batch_size:
                orig_prob, orig_label = model.get_results(batch, batch_size)
                for j in range(batch_size):
                    if orig_label[j] == batch[j][1].item():
                        data = {"code": source_codes[index - batch_size + j + 1], "orig_label": int(orig_label[j])}
                        wf.write(json.dumps(data) + "\n")
                batch = []
                i = 0

dataset = load_dataset(
    'json',
    data_files={
        'test': f"./train_samples.jsonl",
    })

def convert_examples_to_features(js):
    original_code = get_ids(js['code'])
    return {
        "original_code": original_code,
        "label": js['orig_label']
    }

eval_dataset = dataset.map(convert_examples_to_features)
eval_dataset = eval_dataset["test"]

batch_size = 16
train_data = []
for k in tqdm(range(0, len(eval_dataset), batch_size)):
    example = eval_dataset[k:k+batch_size]
    _, _, atts  = model.get_attentions_output(example["original_code"], 6, 16)
    atts = atts[:, 0, :].detach().cpu()

    for i in range(len(example["original_code"])):
        train_data.append((atts[i], example["orig_label"][i]))

torch.save(train_data, "./train_attentions.pt")
    

