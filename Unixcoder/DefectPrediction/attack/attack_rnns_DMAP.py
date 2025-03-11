import sys
import os

sys.path.append('../../../')
sys.path.append('../code')
sys.path.append('../../../algorithms')
sys.path.append('../../../python_parser')
retval = os.getcwd()

import random
import copy
import json
import logging
import argparse
import warnings
import torch
import time
from datetime import datetime

from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, 
                          RobertaTokenizer, RobertaModel,T5Config, T5ForConditionalGeneration)


from model import Model
from run import TextDataset,InputFeatures
from run_parser import get_identifiers, extract_dataflow
from utils import set_seed
from attacker_DMAP import RNNS_Attacker


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
}


logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--train_data_file", default=None, type=str,
                        help="train data file path.")
    parser.add_argument("--valid_data_file", default=None, type=str,
                        help="eval data file path.")  
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="test data file path.")
    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Path to store the CSV file")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    
    parser.add_argument("--model_name", default="unixcoder", type=str,
                        help="The name of model which will be attacked.")


    parser.add_argument("--max_distance", default=0.15, type=float)
    parser.add_argument("--max_length_diff", default=3, type=float)
    parser.add_argument("--substitutes_size", default=60, type=int) 
    parser.add_argument("--iters", default=6, type=int) 
    parser.add_argument("--a", default=0.2, type=float) 
    parser.add_argument("--rnns_type", default="RNNS-Smooth", type=str,  help="") 
    
    
    args = parser.parse_args()

    device = torch.device("cuda")
    args.device = device

    # Set seed
    set_seed(args.seed)

    ## Load Target Model

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    
    config.num_labels=4
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    print('args.block_size',args.block_size) 
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)
    
    model = Model(model, config, tokenizer, args)

    checkpoint_prefix = 'checkpoint-best-f1/%s_model.bin' % args.model_name
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir), False)
    model.to(args.device)
    print("{} - reload model from {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), output_dir))

    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to(args.device)

    ## Load Dataset
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    # Load original source codes
    source_codes = []
    generated_substitutions = []
    with open(args.eval_data_file) as f:
        for line in f:
            js = json.loads(line.strip())
            code = js['func']
            source_codes.append(code)
            generated_substitutions.append(js['substitutes'])
    assert (len(source_codes) == len(eval_dataset) == len(generated_substitutions))
    
    success_attack = 0
    total_cnt = 0
    
    attacker = RNNS_Attacker(args, model, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
    start_time = time.time()
    query_times = 0

    with open(args.csv_store_path, "w") as wf:
        for index, example in enumerate(eval_dataset):
            tmp_save = {"Index":None,"Original Code":None,"Adversarial Code":None,"Program Length":None,"Identifier Num":None,"Replaced Identifiers":None,"Query Times":None,"Time Cost":None,"Type":None}
            print("Index: ", index)
            example_start_time = time.time()
            orig_prob, orig_label = model.get_results([example], args.eval_batch_size)
            orig_label = orig_label[0]
            true_label = example[1].item()
            if not orig_label == true_label:
                continue

            code = source_codes[index]
            _, _, orig_code_tokens = extract_dataflow(code, "c")
            substitutes = generated_substitutions[index]
            identifiers = list(substitutes.keys())
            if len(identifiers) == 0:
                continue

            total_cnt += 1

            code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.attack(example, code)
            attack_type = "rnns"

            example_end_time = (time.time()-example_start_time)/60
            print("Example time cost: ", round(example_end_time, 2), "min")
            print("ALL examples time cost: ", round((time.time()-start_time)/60, 2), "min")
            print("Query times in this attack: ", model.query - query_times)

            replace_info = ''
            if replaced_words is not None:
                for key in replaced_words.keys():
                    replace_info += key + ':' + replaced_words[key] + ','

            if is_success == 1:
                success_attack += 1
                tmp_save["Index"] = index
                tmp_save["Original Code"] = code
                tmp_save["Adversarial Code"] = adv_code
                tmp_save["Program Length"] = len(orig_code_tokens)
                tmp_save["Identifier Num"] = len(identifiers)
                tmp_save["Replaced Identifiers"] = replace_info
                tmp_save["Query Times"] = model.query - query_times
                tmp_save["Time Cost"] = example_end_time
                tmp_save["Type"] = attack_type
            else:
                tmp_save["Index"] = index
                tmp_save["Original Code"] = code
                tmp_save["Adversarial Code"] = adv_code
                tmp_save["Program Length"] = len(orig_code_tokens)
                tmp_save["Identifier Num"] = len(identifiers)
                tmp_save["Query Times"] = model.query - query_times
                tmp_save["Time Cost"] = example_end_time
                tmp_save["Type"] = "0"

            query_times = model.query
            wf.write(json.dumps(tmp_save)+'\n')
            print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


if __name__ == '__main__':
    main()