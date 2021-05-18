'''
Training for the Risk Evaluation Task
'''

import argparse
import yaml
import csv
import logging
import numpy as np
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

logging.basicConfig(level=logging.INFO)


def risk_eval_metrics(eval_pred):
    logits, labels = eval_pred
    scores = np.exp(logits[:, 1]) / \
        (np.exp(logits[:, 0]) + np.exp(logits[:, 1]))
    return {'auroc': roc_auc_score(labels, scores)}


def train(config: dict):
    logging.info('Fine-tuning for the Risk Evalutation Task')
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     config['model']['pretrained'], num_labels=2)
    model = SBertRiskPredictor(config['model']['pretrained'])
    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ClassificationDataset(
            config['data']['train_path'], 'train',
            rand_remove=config['data']['rand_remove']),
        eval_dataset=ClassificationDataset(
            config['data']['train_path'], 'val'),
        compute_metrics=risk_eval_metrics
    )
    trainer.train()

    return trainer


def eval(config: dict, trainer: Trainer):
    logging.info('Evaluating the fine-tuned model')
    for i in range(len(config['data']['test_paths'])):
        logging.info('[{}] Evaluating {}'.format(
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
    parser.add_argument('--config', type=str, help='Path to config')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    set_seed(config['train_args']['seed'])
    trainer = train(config)
    if len(config['data']['test_paths']) > 0:
        eval(config, trainer)
    else:
        logging.info('No testing files found, skipping evaluation...')
