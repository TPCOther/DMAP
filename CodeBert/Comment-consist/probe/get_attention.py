import sys
sys.path.append('../code')

import argparse
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from model import Model
from run import TextDataset
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--block_size", default=510, type=int)

parser.add_argument("--model_name", default="codebert", type=str,
                        help="The name of model which will be attacked.")

args =parser.parse_known_args()[0]
args.device = torch.device("cuda")

config = RobertaConfig.from_pretrained('microsoft/codebert-base')
config.num_labels = 2
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
source_model = RobertaModel.from_pretrained('microsoft/codebert-base', config=config)

model = Model(source_model, config, tokenizer, args)

model.load_state_dict(torch.load("../saved_models/checkpoint-best-f1/codebert_model.bin"), strict=False)
model.to(args.device)
model.eval()

dataset = torch.load("./train_sampled.pt")

for k in range(1, 12):
    train_data = []
    ffn_data = []
    for i in tqdm(range(0, len(dataset), 16)):
        example = []
        for j in range(i, min(i + 16, len(dataset))):
            example.append(dataset[j])
        
        inputs = [[x[0]] for x in example]
        atts, _, _ = model.get_attentions_output(example, 6, 16)
        atts = atts[:, 0, :]
        atts = atts.reshape(-1, atts.size(-1) * 2).detach().cpu()

        for j in range(len(example)):
            train_data.append((atts[j], example[j][1]))

    torch.save(train_data, f"./att/train_attentions_{k}th.pt")