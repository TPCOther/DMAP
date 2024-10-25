import json
import sys
import os

sys.path.append('../../../')
sys.path.append('../code')
sys.path.append('../../../python_parser')
retval = os.getcwd()

import csv
import logging
import argparse
import warnings
import pickle
import copy
import torch
import multiprocessing
import time
import numpy as np
from model import Model
from python_parser.run_parser import get_identifiers
from utils import set_seed, build_vocab
from utils import get_replaced_code, get_code_tokens
from utils import Recorder, is_valid_identifier, CodeDataset
from run import TextDataset
from transformers import (RobertaForMaskedLM, T5Config, T5ForConditionalGeneration, T5EncoderModel, RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

MODEL_CLASSES = {
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
}

logger = logging.getLogger(__name__)

def get_code_pairs(args):
    file_path = args.eval_data_file
    postfix=file_path.split('/')[-1].split('.txt')[0]
    folder = '/'.join(file_path.split('/')[:-1])
    code_pairs_file_path = os.path.join(folder, '{}_cached_{}.pkl'.format(args.model_name, postfix))
    with open(code_pairs_file_path, 'rb') as f:
        code_pairs = pickle.load(f)
    return code_pairs

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--use_ga", action='store_true',
                        help="Whether to GA-Attack.")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    parser.add_argument("--model_name", default="codet5", type=str,
                        help="The name of model which will be attacked.")

    parser.add_argument("--ablation_type", default="origin", type=str)
    
    parser.add_argument("probe_path", type=str, default="../probe/ProbeClassifier.pth")
    parser.add_argument("reserve_ratio", type=int, default=2)
    parser.add_argument("target_layer", type=int, default=6)

    args = parser.parse_args()

    if args.ablation_type == "origin":
        print("Ablation type: ", "origin")
        from attacker_DMAP import WIR_Attacker
    elif args.ablation_type == "wo":
        print("Ablation type: ", "wo")
        from attacker_DMAP_wo import WIR_Attacker
    elif args.ablation_type == "wo_att":
        print("Ablation type: ", "wo_att")
        from attacker_DMAP_wo_att import WIR_Attacker
    elif args.ablation_type == "wo_probe":
        print("Ablation type: ", "wo_probe")
        from attacker_DMAP_wo_probe import WIR_Attacker
    else:
        print("Invalid ablation type")
        exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    args.device = device

    # Set seed
    set_seed(args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
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
    model.load_state_dict(torch.load(output_dir), strict=False)
    model.to(args.device)
    logger.info("reload model from {}".format(output_dir))


    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to(args.device)

    # Load Dataset
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    ## Load code pairs
    source_codes = get_code_pairs(args)

    subs_path = "../dataset/test_subs.jsonl"
    print(subs_path)
    generated_substitutions = []
    with open(subs_path) as f:
        for line in f:
            js = json.loads(line.strip())
            generated_substitutions.append(js["substitutes"])
    assert len(source_codes) == len(eval_dataset) == len(generated_substitutions)

    code_tokens = []
    for index, code in enumerate(source_codes):
        code_tokens.append(get_identifiers(code[2], "java")[1])

    id2token, token2id = build_vocab(code_tokens, 5000)

    success_attack = 0
    total_cnt = 0

    # recoder = Recorder(args.csv_store_path)
    attacker = WIR_Attacker(args, model, tokenizer, token2id, id2token)
    start_time = time.time()
    query_times = 0
    with open(args.csv_store_path, "w") as wf:
        for index, example in enumerate(eval_dataset):
            tmp_save = {"Index":None,"Original Code":None, "Sucpre Code": None,"Adversarial Code":None,"Program Length":None,"Identifier Num":None,"Replaced Identifiers":None,"Query Times":None,"Time Cost":None,"Type":None}
            print("Index: ", index)
            example_start_time = time.time()
            code_pair = source_codes[index]
            logits, preds = model.get_results([example], args.eval_batch_size)
            orig_label = preds[0]
            true_label = example[1].item()
            if not orig_label == true_label:
                continue

            substitutes = generated_substitutions[index]
            first_code = code_pair[2]

            code_2 = code_pair[3]
            code_2 = " ".join(code_2.split())
            words_2 = tokenizer.tokenize(code_2)

            orig_code_tokens = get_code_tokens(first_code)
            identifiers = list(substitutes.keys())
            if len(identifiers) == 0:
                continue
            total_cnt += 1

            code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.wir_random_attack(
                example, code_pair)

            example_end_time = (time.time() - example_start_time) / 60
            print("Example time cost: ", round(example_end_time, 2), "min")
            print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
            print("Query times in this attack: ", model.query - query_times)

            replace_info = ''
            if replaced_words is not None:
                for key in replaced_words.keys():
                    replace_info += key + ':' + replaced_words[key] + ','

            if is_success == 1:
                success_attack += 1
                tmp_save["Index"] = index
                tmp_save["Original Code"] = code_pair[2]
                tmp_save["Adversarial Code"] = adv_code
                tmp_save["Program Length"] = len(orig_code_tokens)
                tmp_save["Identifier Num"] = len(identifiers)
                tmp_save["Replaced Identifiers"] = replace_info
                tmp_save["Query Times"] = model.query - query_times
                tmp_save["Time Cost"] = example_end_time
                tmp_save["Type"] = "wir"

            else:
                tmp_save["Index"] = index
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