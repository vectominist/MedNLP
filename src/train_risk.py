'''
Training for the Risk Evaluation Task
'''

import argparse
import yaml
import os
import csv
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    set_seed
)
from data import ClassificationDataset
from model.risk_model_sbert import SBertRiskPredictor


def risk_eval_metrics(eval_pred):
    logits, labels = eval_pred
    scores = np.exp(logits[:, 1]) / \
        (np.exp(logits[:, 0]) + np.exp(logits[:, 1]))
    return {'auroc': roc_auc_score(labels, scores)}


def train(config: dict, val: bool = True):
    print('Fine-tuning for the Risk Evalutation Task')
    model = SBertRiskPredictor(config['model']['pretrained'])
    tr_set = ClassificationDataset(
        config['data']['train_path'], 'train',
        val_r=10 if val else 10000,
        rand_remove=config['data']['rand_remove'],
        rand_swap=config['data']['rand_swap'],
        eda=config['data']['eda'])
    dv_set = None if not val else \
        ClassificationDataset(config['data']['train_path'], 'val')
    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tr_set,
        eval_dataset=dv_set,
        compute_metrics=risk_eval_metrics
    )
    trainer.train()


def eval(config: dict, ckpt: str):
    print('Evaluating the fine-tuned model')
    print('Loading model from {}'.format(ckpt))
    model = SBertRiskPredictor(config['model']['pretrained'])
    ckpt = torch.load(os.path.join(ckpt, 'pytorch_model.bin'))
    model.load_state_dict(ckpt)

    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(model=model, args=training_args)

    for i in range(len(config['data']['test_paths'])):
        print('[{}] Evaluating {}'.format(
            i + 1, config['data']['test_paths'][i]))

        tt_set = ClassificationDataset(config['data']['test_paths'][i], 'test')
        logits, _, _ = trainer.predict(tt_set)
        scores = np.exp(logits[:, 1]) / \
            (np.exp(logits[:, 0]) + np.exp(logits[:, 1]))

        with open(config['data']['pred_paths'][i], 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['article_id', 'probability'])
            ids = tt_set.get_ids()
            assert len(ids) == len(scores)
            for j in range(len(scores)):
                writer.writerow([str(ids[j]), scores[j]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training for Risk Evaluation')
    parser.add_argument('--mode', choices=['train', 'test'], help='Mode')
    parser.add_argument('--config', type=str, help='Path to config')
    parser.add_argument('--ckpt', type=str, default='', help='Path to ckpt')
    parser.add_argument('--val', action='set_true', help='Use val set')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    set_seed(config['train_args']['seed'])

    if args.mode == 'train':
        train(config, val=args.val)
    else:
        eval(config, args.ckpt)
