# CodeBert
This Directory provide the code for the AdvSel attack on CodeBert model.

## 📁 Directory Structure
    CodeT5
    ├── Clone-detection                 
    │   ├── attack
    │   │   ├── attack_{method}.py          # original attack script
    │   │   ├── attack_{method}_DMAP.py     # attack script with AdvSel
    │   │   ├── attacker.py                 # original attack code
    │   │   ├── attacker_DMAP.py            # attack code with AdvSel
    │   │   ├── attacker_DMAP_{ablation}.py # ablation attacker
    │   │   ├── DMAP.py                     # AdvSel method code
    │   │   └── DMAP_{ablation}.py          # ablation AdvSel
    │   ├── code
    │   │   ├── model.py                    # model running code
    │   │   └── run.py                      # code for data processing
    │   ├── probe
    │   │   ├── get_attention.py     # code for getting trainning data of probe
    │   │   └── probe.ipynb          # code for training probe
    │   ├── saved_models
    │   └── dataset           
    ├── DefectPrediction
    └── Vulnerability-prediction

## 📚 Training Probe

To train the probe model, you need to run the following command to get training data of probe:

```bash
python get_attention.py
```

Then just run the code in the probe.ipynb file and save probe.

## 🚀 Running experiments

Run the following command to attack the CodeBert model:

### Alert:
For Clone-detection task:
```bash
python attack_alert.py \
    --output_dir=../saved_models \
    --eval_data_file=../dataset/test_sampled.txt \
    --subs_path=../dataset/test_subs.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --model_type=codet5 \
    --model_name_or_path=Salesforce/codet5-base-multi-sum \
    --csv_store_path=./result/attack_alert_all.jsonl \
    --config_name=Salesforce/codet5-base-multi-sum \
    --tokenizer_name=Salesforce/codet5-base-multi-sum \
    --use_ga \
    --eval_batch_size 16 \
    --seed 123456
```
For Other tasks:
```bash
python attack_alert.py \
        --output_dir=../saved_models \
        --model_type=codet5 \
        --tokenizer_name=Salesforce/codet5-base-multi-sum \
        --model_name_or_path=Salesforce/codet5-base-multi-sum \
        --csv_store_path result/attack_alert_all.json \
        --base_model=microsoft/codebert-base-mlm \
        --eval_data_file=../dataset/test_subs.jsonl \
        --use_ga \
        --block_size 512 \
        --eval_batch_size 16 \
        --seed 123456
```

### RNNS:
For Clone-detection task:
```bash
python attack_rnns.py \
    --output_dir=../saved_models/ \
    --model_type=codet5 \
    --config_name=Salesforce/codet5-base-multi-sum \
    --model_name_or_path=Salesforce/codet5-base-multi-sum \
    --tokenizer_name=Salesforce/codet5-base-multi-sum \
    --base_model=microsoft/codebert-base-mlm \
    --max_distance=0.15 \
    --max_length_diff=4 \
    --substitutes_size=60  \
    --iters=6 \
    --a=0.2 \
    --csv_store_path result/attack_rnns_all.jsonl \
    --eval_data_file=../dataset/test_sample.txt \
    --train_data_file=../dataset/train_sampled.txt \
    --valid_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 16 \
    --seed 123456
```
For Other tasks:
```bash
python attack_rnns.py \
        --output_dir=../saved_models/ \
        --model_type=codet5 \
        --base_model_type=roberta \
        --config_name=Salesforce/codet5-base-multi-sum \
        --model_name_or_path=Salesforce/codet5-base-multi-sum \
        --tokenizer_name=Salesforce/codet5-base-multi-sum \
        --base_model=microsoft/codebert-base-mlm \
        --max_distance=0.15 \
        --max_length_diff=3 \
        --substitutes_size=60  \
        --iters=6 \
        --a=0.2 \
        --csv_store_path result/attack_rnns_all.jsonl \
        --eval_data_file=../dataset/test_subs.jsonl \
        --train_data_file=../dataset/train.jsonl \
        --valid_data_file=../dataset/test.jsonl \
        --test_data_file=../dataset/test.jsonl \
        --block_size 512 \
        --eval_batch_size 32 \
        --seed 123456
```

### WIR:
For Clone-detection task:
```bash
python attack_wir.py \
    --output_dir=../saved_models \
    --model_type=codet5 \
    --tokenizer_name=Salesforce/codet5-base-multi-sum \
    --model_name_or_path=Salesforce/codet5-base-multi-sum \
    --csv_store_path result/attack_wir_all.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=../dataset/test_sample.txt \
    --block_size 512 \
    --eval_batch_size 16 \
    --seed 123456
```
For Other tasks:
```bash
python attack_wir.py \
        --output_dir=../saved_models \
        --model_type=codet5 \
        --config_name=Salesforce/codet5-base-multi-sum \
        --tokenizer_name=Salesforce/codet5-base-multi-sum \
        --model_name_or_path=Salesforce/codet5-base-multi-sum \
        --csv_store_path result/attack_wir_all.jsonl \
        --base_model=microsoft/codebert-base-mlm \
        --eval_data_file=../dataset/test_subs.jsonl \
        --block_size 512 \
        --eval_batch_size 32 \
        --seed 123456
```

If you want to run the attack with AdvSel, just change the script name to attack_{method}_DMAP.py. For example:

```bash
python attack_alert_DMAP.py \
        --output_dir=../saved_models \
        --model_type=codet5 \
        --tokenizer_name=Salesforce/codet5-base-multi-sum \
        --model_name_or_path=Salesforce/codet5-base-multi-sum \
        --csv_store_path result/attack_alert_all.json \
        --base_model=microsoft/codebert-base-mlm \
        --eval_data_file=../dataset/test_subs.jsonl \
        --use_ga \
        --block_size 512 \
        --eval_batch_size 16 \
        --seed 123456
```

## 🚀 Running Ablation

Run the following command to run ablation, change ablation_type(wo(Random), wo_att(APM), wo_probe(DDPM)) to the ablation you want to run:

```bash
python attack_alert_DMAP.py \
        --output_dir=../saved_models \
        --model_type=codet5 \
        --tokenizer_name=Salesforce/codet5-base-multi-sum \
        --model_name_or_path=Salesforce/codet5-base-multi-sum \
        --csv_store_path result/attack_alert_DMAP.json \
        --base_model=microsoft/codebert-base-mlm \
        --eval_data_file=../dataset/test_subs_mix_cb.jsonl \
        --use_ga \
        --block_size 512 \
        --eval_batch_size 16 \
        --ablation_type={ablation} \
        --seed 123456
```