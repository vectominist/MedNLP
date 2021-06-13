#!/bin/bash

# python3 src/svc.py \
#     --train data/Train_risk_classification_ans.csv \
#     --val data/Develop_risk_classification_ans.csv \
#     --test data/Develop_risk_classification_ans.csv \
#     --out data/decision.csv

python3 src/svc.py \
    --train data/train_risk_tr-dv.csv \
    --val data/Develop_risk_classification_ans.csv \
    --test data/Develop_risk_classification_ans.csv \
    --out data/decision.csv
