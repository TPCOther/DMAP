import sys
sys.path.append('../code')
sys.path.append('../../../python_parser')
from model import Model
from run import TextDataset
from run_parser import extract_dataflow
from torch.utils.data.dataset import Dataset
import numpy as np
from tqdm import tqdm
import json
import argparse
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from datasets import load_dataset
import torch

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg

class GraphCodeDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args=args
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx))


def get_cosine_similarity(a, b):
    a = a.squeeze(0).mean(dim=0)
    b = b.squeeze(0).mean(dim=0)
    dot_product = torch.sum(a * b)
    norm_A = torch.sqrt(torch.sum(a * a))
    norm_B = torch.sqrt(torch.sum(b * b))
    return dot_product / (norm_A * norm_B)

parser = argparse.ArgumentParser()

parser.add_argument("--block_size", default=510, type=int)

parser.add_argument("--model_name", default="graphcodebert", type=str,
                        help="The name of model which will be attacked.")

parser.add_argument("--data_flow_length", default=64, type=int,
                    help="Optional Data Flow input sequence length after tokenization.") 
parser.add_argument("--code_length", default=256, type=int,
                    help="Optional Code input sequence length after tokenization.") 

args =parser.parse_known_args()[0]
args.device = torch.device("cuda")

config = RobertaConfig.from_pretrained('microsoft/graphcodebert-base')
config.num_labels = 4
tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
source_model = RobertaForSequenceClassification.from_pretrained('microsoft/graphcodebert-base', config=config)

model = Model(source_model, config, tokenizer, args)

model.load_state_dict(torch.load("../saved_models/checkpoint-best-f1/graphcodebert_model.bin"), strict=False)
model.to(args.device)
model.eval()

def convert_examples_to_features(js):
    #source
    code=' '.join(js.split())
    dfg, index_table, code_tokens = extract_dataflow(code, "c")

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

    return InputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg)

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

with open("./train_samples.jsonl", "w") as wf:
        for index, example in enumerate(tqdm(eval_dataset)):
            orig_prob, orig_label = model.get_results([example], 2)
            if orig_label[0] == example[3].item():
                data = {"code": source_codes[index], "orig_label": int(orig_label[0])}
                wf.write(json.dumps(data) + "\n")

dataset = load_dataset(
    'json',
    data_files={
        'test': "./train_samples.jsonl"
    })

eval_dataset = dataset["test"]

train_data = []
batch_size = 16
for k in tqdm(range(0, len(eval_dataset), batch_size)):
    example = eval_dataset[k:k+batch_size]
    codeset = GraphCodeDataset([convert_examples_to_features(x) for x in example["code"]], args)

    _, _, atts = model.get_attentions_output(codeset, 6, 16)
    atts = atts[:, 0, :].detach().cpu()

    for j in range(len(example["code"])):
        train_data.append((atts[j], example["orig_label"][j]))

torch.save(train_data, "./train_attentions.pt")

