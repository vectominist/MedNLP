'''
    Training for Multitask Learning
'''

import argparse
import yaml
import os
import csv
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import (
    TrainingArguments,
    Trainer,
    EvalPrediction,
    set_seed
)
from data import MultiTaskDataset, ClassificationDataset, QADataset2
from model.joint_model_sbert import SBertJointPredictor
from train_risk import risk_eval_metrics

label2answer = ['A', 'B', 'C']


def qa_eval_metrics(eval_pred):
    logits, labels = eval_pred
    answer = np.argmax(logits, axis=1)
    accuracy = (answer == labels).astype(float).mean()
    return {'acc': accuracy}


def train(config: dict, val: bool = True, val_mode: str = 'risk'):
    print('MTL fine-tuning')
    model = SBertJointPredictor(config['model']['pretrained'])
    model.eval_qa(val_mode == 'qa')
    tr_set = MultiTaskDataset(
        config['data']['train_path_risk'],
        config['data']['train_path_qa'],
        'train',
        val_r=10 if val else 10000,
        rand_remove=config['data']['rand_remove'],
        rand_swap=config['data']['rand_swap'],
        eda=config['data']['eda']
    )
    dv_set = None if not val else MultiTaskDataset(
        config['data']['train_path_risk'],
        config['data']['train_path_qa'],
        'val',
        val_r=10,
        val_mode=val_mode
    )
    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tr_set,
        eval_dataset=dv_set,
        compute_metrics=risk_eval_metrics if val_mode == 'risk' else qa_eval_metrics
    )
    trainer.train()


def eval(config: dict, ckpt: str):
    print('Evaluating the fine-tuned model')

    print('Loading model from {}'.format(ckpt))
    model = SBertJointPredictor(config['model']['pretrained'])
    ckpt = torch.load(os.path.join(ckpt, 'pytorch_model.bin'))
    model.load_state_dict(ckpt)

    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(model=model, args=training_args)

    # Validation
    print('Validation for Risk Prediction')
    dv_set = ClassificationDataset(config['data']['train_path_risk'], 'val')
    trainer.compute_metrics = risk_eval_metrics
    metric = trainer.evaluate(dv_set)
    print('Val AUROC = {:.3f}'.format(metric['eval_auroc']))

    print('Validation for QA')
    dv_set = QADataset2(config['data']['train_path_qa'], 'val')
    trainer.compute_metrics = qa_eval_metrics
    trainer.model.eval_qa(True)
    metric = trainer.evaluate(dv_set)
    print('Val ACC = {:.1f}%'.format(metric['eval_acc'] * 100.))

    # Evaluate Risk Prediction
    print('Evaluation for Risk Prediction')
    trainer.model.eval_qa(False)
    for i in range(len(config['data']['test_paths_risk'])):
        print('[{}] Evaluating {}'.format(
            i + 1, config['data']['test_paths_risk'][i]))

        tt_set = ClassificationDataset(
            config['data']['test_paths_risk'][i], 'test')
        logits, _, _ = trainer.predict(tt_set)
        scores = np.exp(logits[:, 1]) / \
            (np.exp(logits[:, 0]) + np.exp(logits[:, 1]))

        with open(config['data']['pred_paths_risk'][i], 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['article_id', 'probability'])
            ids = tt_set.get_ids()
            assert len(ids) == len(scores)
            for j in range(len(scores)):
                writer.writerow([str(ids[j]), scores[j]])

    # Evaluate QA
    print('Evaluation for QA')
    trainer.model.eval_qa(True)
    for i in range(len(config['data']['test_paths_qa'])):
        print('[{}] Evaluating {}'.format(
            i + 1, config['data']['test_paths_qa'][i]))

        tt_set = QADataset2(config['data']['test_paths_qa'][i], 'test')
        logits, _, _ = trainer.predict(tt_set)
        answers = np.argmax(logits, axis=1)

        with open(config['data']['pred_paths_qa'][i], 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'answer'])
            ids = tt_set.get_ids()
            assert len(ids) == len(answers)
            for j in range(len(answers)):
                writer.writerow([str(ids[j]), label2answer[answers[j]]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training for Risk Evaluation')
    parser.add_argument('--mode', choices=['train', 'test'], help='Mode')
    parser.add_argument('--config', type=str, help='Path to config')
    parser.add_argument('--ckpt', type=str, default='', help='Path to ckpt')
    parser.add_argument('--val', action='store_true', help='Use val set')
    parser.add_argument('--val-risk', action='store_true', help='Use val set (risk)')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    set_seed(config['train_args']['seed'])

    if args.mode == 'train':
        train(config, val=args.val, val_mode='risk' if args.val_risk else 'qa')
    else:
        eval(config, args.ckpt)
