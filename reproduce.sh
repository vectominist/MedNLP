#!/bin/bash
# This is the script to reproduce our results for the AI cup competition.

risk_config=$1
qa_config=$2
out_dir=output

mkdir -p $out_dir

# === Risk ===
echo "== Reproducing the risk assessment task =="
echo ""

# DL-based Method
echo "DL-based 1"
python3 src/train_risk.py \
    --mode test \
    --config $risk_config \
    --ckpt checkpoints/sbert_ckpt/tr-dv-7122 \
    --out $out_dir/decision.bert.1.csv

echo "DL-based 2"
python3 src/train_risk.py \
    --mode test \
    --config $risk_config \
    --ckpt checkpoints/sbert_ckpt/tr-dv-8888 \
    --out $out_dir/decision.bert.2.csv

# Gradient Boosting
echo "Gradient Boosting 1"
python3 src/tfidf_test.py \
    --test data/Develop_risk_classification.csv \
    --ckpt checkpoints/gdboost_ckpt/tr-dv-7122 \
    --out $out_dir/decision.gdboost.1.csv \
    --seed 7122

echo "Gradient Boosting 2"
python3 src/tfidf_test.py \
    --test data/Develop_risk_classification.csv \
    --ckpt checkpoints/gdboost_ckpt/tr-dv-8888 \
    --out $out_dir/decision.gdboost.2.csv \
    --seed 8888

python3 src/util/ensemble_risk_pred.py \
    --preds $out_dir/decision.bert.1.csv $out_dir/decision.bert.2.csv \
            $out_dir/decision.gdboost.1.csv $out_dir/decision.gdboost.2.csv \
    --out $out_dir/decision.csv

rm -f $out_dir/decision.bert.1.csv $out_dir/decision.bert.2.csv 
rm -f $out_dir/decision.gdboost.1.csv $out_dir/decision.gdboost.2.csv
echo ""


# === QA ===
echo "== Reproducing the qa task =="
echo ""
python3 src/qa_rule_base.py --config $qa_config

echo ""


# zip file
zip -r results.zip $out_dir

echo "Finished reproducing, results are compressed in results.zip"
