'''
Training for the Risk Evaluation Task
'''

import argparse
import yaml
import os
import csv
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
# from data import ClassificationDataset
from data_doc2vec import ClassificationDataset
# from model.risk_model_sbert import SBertRiskPredictor
from model.risk_model_doc2vec import Doc2VecRiskPredictor


def risk_eval_metrics(eval_pred):
    logits, labels = eval_pred
    scores = np.exp(logits[:, 1]) / \
        (np.exp(logits[:, 0]) + np.exp(logits[:, 1]))
    return {'auroc': roc_auc_score(labels, scores)}


def train(config: dict):
    print('Fine-tuning for the Risk Evalutation Task')
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     config['model']['pretrained'], num_labels=2)
    # model = SBertRiskPredictor(config['model']['pretrained'])
    model = Doc2VecRiskPredictor(
        config['model']['att_dim'], config['model']['doc_dim'])
    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ClassificationDataset(
            config['data']['train_path'], 'train',
            rand_remove=config['data']['rand_remove'],
            doc2vec=config['model']['doc2vec']),
        eval_dataset=ClassificationDataset(
            config['data']['train_path'], 'val',
            doc2vec=config['model']['doc2vec']),
        compute_metrics=risk_eval_metrics
    )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=ClassificationDataset(
    #         config['data']['train_path'], 'train',
    #         rand_remove=config['data']['rand_remove']),
    #     eval_dataset=ClassificationDataset(
    #         config['data']['train_path'], 'val'),
    #     compute_metrics=risk_eval_metrics
    # )
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
                writer.writerow([str(ids[j]), scores[i]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training for Risk Evaluation')
    parser.add_argument(
        '--mode', choices=['train', 'test'], help='Mode for risk eval task')
    parser.add_argument('--config', type=str, help='Path to config')
    parser.add_argument(
        '--ckpt', type=str, default='../runs/risk-1/checkpoint-44', help='Path to checkpoint')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    set_seed(config['train_args']['seed'])

    if args.mode == 'train':
        train(config)
    else:
        eval(config, args.ckpt)
