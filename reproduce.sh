#!/bin/bash
# This is the script to reproduce our results for the AI cup competition.

[ $# -ne 2 ] && echo "Usage: $0 <risk config> <qa config>" && exit 1

risk_config=$1
qa_config=$2
out_dir=output

sec=$SECONDS

mkdir -p $out_dir

if ! command -v yq &> /dev/null
then
	echo "yq not found! Run one of the following command."
	echo "sudo apt-get install yq"
	echo "conda install yq"
	exit 1
fi

risk_test=`yq .data.test_paths[0] $risk_config | tr -d '"'`

# === Risk ===
echo "== Reproducing the risk assessment task =="
echo ""

# DL-based Method
echo "DL-based 1"
python3 src/train_risk.py \
    --mode test \
    --config $risk_config \
    --ckpt checkpoints/sbert_ckpt/tr-dv-7122 \
    --out $out_dir/decision.bert.1.csv || exit 1

echo "DL-based 2"
python3 src/train_risk.py \
    --mode test \
    --config $risk_config \
    --ckpt checkpoints/sbert_ckpt/tr-dv-8888 \
    --out $out_dir/decision.bert.2.csv || exit 1

# Gradient Boosting
echo "Gradient Boosting 1"
python3 src/tfidf_test.py \
    --test ${risk_test} \
    --ckpt checkpoints/gdboost_ckpt/tr-dv-7122 \
    --out $out_dir/decision.gdboost.1.csv \
    --seed 7122 || exit 1

echo "Gradient Boosting 2"
python3 src/tfidf_test.py \
    --test ${risk_test} \
    --ckpt checkpoints/gdboost_ckpt/tr-dv-8888 \
    --out $out_dir/decision.gdboost.2.csv \
    --seed 8888 || exit 1

python3 src/util/ensemble_risk_pred.py \
    --preds $out_dir/decision.bert.1.csv $out_dir/decision.bert.2.csv \
            $out_dir/decision.gdboost.1.csv $out_dir/decision.gdboost.2.csv \
    --out $out_dir/decision.csv || exit 1

rm -f $out_dir/decision.bert.1.csv $out_dir/decision.bert.2.csv 
rm -f $out_dir/decision.gdboost.1.csv $out_dir/decision.gdboost.2.csv
echo ""


# === QA ===
echo "== Reproducing the qa task =="
echo ""
python3 src/qa_rule_base.py --config $qa_config || exit 1

echo ""


# zip file
zip -rj results.zip $out_dir/*

echo "Finished reproducing, results are compressed in results.zip"

echo "Checking format..."
python check_format.py || exit 1
rm -rf qa.csv decision.csv
echo ""
echo "=========================================="
echo "Total run time `./timer.pl $[$SECONDS-$sec]`"
echo "=========================================="

