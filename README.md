# Cost-Effective Adversarial Attacks Against Pre-Trained Code Models with Model Attention(AdvSel)

## ✨ Overview

This is the codebase for the paper "Cost-Effective Adversarial Attacks Against Pre-Trained Code Models with Model Attention". We propose AdvSel, a novel framework that enhances attack efficiency while maintaining effectiveness. Designed as a plug-and-play solution, AdvSel integrates seamlessly with existing attack techniques. It generates a large pool of mutated samples using current attack methods and then employs two key components—the Attention Proxy Model (APM) and the Deviation Direction Proxy Model (DDPM)—to identify suspicious samples more likely to mislead the model. By filtering out unlikely mutation candidates, AdvSel significantly reduces the need for frequent queries to the victim model. 

## 📁 Directory Structure

    .
    ├── CodeBert                    # Attack code for every model
    │   ├── Clone-detection         # Attack code for every task
    │   │   ├── attack              # Main function of attack
    │   │   ├── code                # Code for model running
    │   │   ├── probe               # Code of deviation direction proxy model
    │   │   ├── saved_models        # Victim pre-trained models
    │   │   └── dataset             # Dataset for each task
    │   ├── DefectPrediction
    │   ├── Comment-consist
    │   └── Vulnerability-prediction
    ├── CodeT5
    ├── GraphCodeBert
    ├── PLBART
    ├── Unixcoder
    ├── python_parser               # Code for parsing code samples
    ├── EmpiricalResult             # Statistics from empirical studies
    └── RQ5_Result                  # Results of limiting the number of queries            

## 🔨 Setup environment
- Prerequisite:
    - Install conda package manager
- Create python environment with required dependencies, The concrete enviroment requirments are put in the `attack.yaml` file.
    ```bash
    conda env create -f attack.yaml
    ```
- We have to compress large files in order to upload them to internert.
    - Download the compressed files from [zenodo](https://zenodo.org/records/13989676?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjAxOWNiYTllLTczODAtNDAxMy05ZTFjLWU1YzdjMDRmMTZjMSIsImRhdGEiOnt9LCJyYW5kb20iOiJiMjgwOGQ4N2MxOGUyNGMyZGRkNWVmMjIyZWVmZTRjMyJ9.7GQ3bQZgfcpxiRohaYUNdOjoS2-rNW2__rxvJA7IFZdJDr4d_pH-chLwtwj4fy43QMlFQS_UmVvHwuddNc2jKw)
    - Unzip the files and put them in the corresponding directories before running code.


## 🚀 Running experiments
See README.md in each model directory for more details.

## 🚀 Running Ablation
See README.md in CodeT5 model directory for more details.