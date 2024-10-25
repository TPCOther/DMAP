import sys
sys.path.append('../code')

import argparse
from transformers import T5Config, AutoTokenizer, T5ForConditionalGeneration
from model import Model
from run import TextDataset
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--code_length", default=256, type=int,
                    help="Optional Code input sequence length after tokenization.") 
parser.add_argument("--data_flow_length", default=64, type=int,
                    help="Optional Data Flow input sequence length after tokenization.")
parser.add_argument("--block_size", default=510, type=int)

parser.add_argument("--model_name", default="graphcodebert", type=str,
                        help="The name of model which will be attacked.")

args =parser.parse_known_args()[0]
args.device = torch.device("cuda")

config = T5Config.from_pretrained('Salesforce/codet5-base-multi-sum')
config.num_labels = 2
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum', config=config)

model = Model(model, config, tokenizer, args)

model.load_state_dict(torch.load("../saved_models/checkpoint-best-f1/codet5_model.bin"), strict=False)
model.to(args.device)
model.eval()


eval_dataset = TextDataset(tokenizer, args, "../dataset/real_train_sampled.txt")

batch_size = 16
correct_data = []
print(len(eval_dataset))
for i in tqdm(range(0, len(eval_dataset), batch_size)):
    example = []
    for j in range(i, min(i + batch_size, len(eval_dataset))):
        example.append(eval_dataset[j])
    
    
    logits, preds = model.get_results(example, 16)

    for j in range(len(example)):
        orig_label = preds[j]
        true_label = example[j][1].item()
        if not orig_label == true_label:
            continue
        else:
            correct_data.append(example[j])

dataset = correct_data

train_data = []
for i in tqdm(range(0, len(dataset), 16)):
    example = []
    for j in range(i, min(i + 16, len(dataset))):
        example.append(dataset[j])
    
    inputs = [[x[0]] for x in example]
    _, _, _, atts = model.get_attentions_output(example, 6, 16)
    atts = atts[:, 0, :]
    atts = atts.reshape(-1, atts.size(-1) * 2).detach().cpu()

    for j in range(len(example)):
        train_data.append((atts[j], example[j][1]))

torch.save(train_data, "./train_attentions.pt")
    
