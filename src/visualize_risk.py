'''
    File      [ src/visualize_risk.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Visualization (ROC curve) for the risk task ]
'''

import argparse
import yaml
import os
import numpy as np
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed
)
from data import ClassificationDataset
from model.risk_model_sbert import SBertRiskPredictor

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import logging
logging.disable(logging.WARNING)


def eval(config: dict, ckpt: str, data_path: str):
    print('Evaluating the fine-tuned model')
    print('Loading model from {}'.format(ckpt))
    model = SBertRiskPredictor(**config['model'])
    ckpt = torch.load(os.path.join(ckpt, 'pytorch_model.bin'))
    model.load_state_dict(ckpt)

    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(model=model, args=training_args)

    def pred(data):
        dataset = ClassificationDataset(data, 'train', val_r=10000)
        labels = np.array([d[2] for d in dataset.data])
        logits, _, _ = trainer.predict(dataset)
        scores = np.exp(logits[:, 1]) / \
            (np.exp(logits[:, 0]) + np.exp(logits[:, 1]))
        auc = roc_auc_score(labels, scores)
        return labels, scores, auc

    tr_labels, tr_scores, tr_auc = pred(config['data']['train_path'])
    dv_labels, dv_scores, dv_auc = pred(data_path)

    print()
    print('Train AUC = {:.5f}'.format(tr_auc))
    print('Dev   AUC = {:.5f}'.format(dv_auc))

    fpr, tpr, thresholds = roc_curve(dv_labels, dv_scores)

    lw = 2
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % dv_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/roc.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualization for Risk Assessment')
    parser.add_argument('--config', type=str, help='Path to config')
    parser.add_argument('--ckpt', type=str, default='', help='Path to ckpt')
    parser.add_argument('--data', type=str, default='', help='Path to data')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    set_seed(config['train_args']['seed'])

    eval(config, args.ckpt, args.data)
