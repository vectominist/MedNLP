#!/bin/bash
# This is the script to reproduce our results for the AI cup competition.

risk_config=$1
qa_config=$2
out_dir=$3

# === Risk ===
echo "Reproducing the risk assessment task"
# Download model
mkdir -p model
gdown --id "" --output model/pretrained.tgz
tar zxvf model/pretrained.tgz

# DL-based Method
python3 src/train_risk.py \
    --mode "eval" \
    --config $risk_config \
    --ckpt "" \
    --out $out_dir/decision.bert.csv

# Gradient Boosting
python3 src/svc.py \
    --train material/ \
    --val material/ \
    --test material/ \
    --out $out_dir/decision.gradboost.csv

python3 src/util/ensemble_risk_pred.py \
    --preds $out_dir/decision.bert.csv $out_dir/decision.gradboost.csv
    --out $out_dir/decision.csv

rm -f $out_dir/decision.bert.csv $out_dir/decision.gradboost.csv


# === QA ===
echo "Reproducing the qa task"
python3 src/qa_rule_base.py --config $qa_config


# zip file
zip -r results.zip output

echo "Finished reproducing, results are compressed in results.zip"
