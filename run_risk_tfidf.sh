#!/bin/bash

# train set

python3 src/tfidf.py \
    --train data/Train_risk_classification_ans.csv \
    --val data/Develop_risk_classification_ans.csv \
    --ckpt checkpoints/gdboost_ckpt/tr-7122 \
    --cls gdboost \
    --seed 7122

# python3 src/tfidf_test.py \
#     --test data/Develop_risk_classification_ans.csv \
#     --ckpt checkpoints/gdboost_ckpt/tr-7122 \
#     --out data/decision.gdboost.tr.csv \
#     --seed 7122

# train + dev sets
# python3 src/tfidf.py \
#     --train data/train_risk_tr-dv.csv \
#     --val data/Develop_risk_classification_ans.csv \
#     --ckpt checkpoints/gdboost_ckpt/tr-dv-7122 \
#     --cls gdboost \
#     --seed 7122

# python3 src/tfidf_test.py \
#     --test data/Develop_risk_classification_ans.csv \
#     --ckpt checkpoints/gdboost_ckpt/tr-dv-7122 \
#     --out data/decision.gdboost.tr-dv.csv \
#     --seed 7122

# Grid search
# python3 src/gdboost_grid_search.py \
#     --train data/Train_risk_classification_ans.csv \
#     --val data/Develop_risk_classification_ans.csv \
#     --seed 7122
