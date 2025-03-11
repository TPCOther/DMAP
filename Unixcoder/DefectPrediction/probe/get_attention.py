import sys
sys.path.append('../code')
sys.path.append('../attack')
import argparse
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from model import Model
from run import TextDataset
from tqdm import tqdm
import torch
from datasets import load_dataset

parser = argparse.ArgumentParser()

parser.add_argument("--block_size", default=510, type=int)

parser.add_argument("--model_name", default="codebert", type=str,
                        help="The name of model which will be attacked.")

args =parser.parse_known_args()[0]
args.device = torch.device("cuda")

config = RobertaConfig.from_pretrained('microsoft/unixcoder-base')
config.num_labels = 4
tokenizer = RobertaTokenizer.from_pretrained('microsoft/unixcoder-base')
source_model = RobertaModel.from_pretrained('microsoft/unixcoder-base', config=config)

model = Model(source_model, config, tokenizer, args)

model.load_state_dict(torch.load("../saved_models/checkpoint-best-f1/unixcoder_model.bin"), strict=False)
model.to(args.device)
model.eval()

def get_ids(code):
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-4]
    source_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return torch.tensor(source_ids)

dataset = load_dataset(
    'json',
    data_files={
        'test': f"./train_samples_new.jsonl",
    })

def convert_examples_to_features(js):
    original_code = get_ids(js['code'])
    return {
        "original_code": original_code,
        "label": js['orig_label']
    }

eval_dataset = dataset.map(convert_examples_to_features)
eval_dataset = eval_dataset["test"]

for m in range(1, 16):
    batch_size = 16
    train_data = []
    ffns_data = []
    for k in tqdm(range(0, len(eval_dataset), batch_size)):
        example = eval_dataset[k:k+batch_size]
        ffns, _, atts  = model.get_attentions_output(example["original_code"], 9, 16)
        atts = atts[:, 0, :].detach().cpu()
        ffns = ffns[:, 0, :].detach().cpu()

        for i in range(len(example["original_code"])):
            train_data.append((atts[i], example["orig_label"][i]))
            ffns_data.append((ffns[i], example["orig_label"][i]))

    torch.save(train_data, f"./att/train_attentions_{m}th.pt")
    torch.save(ffns_data, f"./ffn/train_attentions_{m}th.pt")