'''
    Training for the Multichoice QA
'''

import argparse
import yaml
import os
import csv
import numpy as np
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    EvalPrediction,
    set_seed
)
from data import QADataset3
from model.qa_model_sbert import SBertQA


label2answer = ['A', 'B', 'C']


def qa_eval_metrics(eval_pred):
    logits, labels = eval_pred
    answer = np.argmax(logits, axis=1)
    accuracy = (answer == labels).astype(float).mean()
    return {'acc': accuracy}


def train(config: dict, val: bool = True):
    print('Fine-tuning for the Risk Evalutation Task')
    model = SBertQA(config['model']['pretrained'])
    tr_set = QADataset3(
        config['data']['train_path'], 'train',
        val_r=10000,  # FIXME: ignore val
        rand_remove=config['data']['rand_remove'],
        rand_swap=config['data']['rand_swap'],
        eda=config['data']['eda'],
        doc_splits=config['data'].get('doc_splits', 1))
    dv_set = QADataset3(
        config['data']['val_path'], 'train', val_r=10000,
        doc_splits=config['data'].get('doc_splits', 1))
    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tr_set,
        eval_dataset=dv_set,
        compute_metrics=qa_eval_metrics
    )
    trainer.train()


def eval(config: dict, ckpt: str):
    print('Evaluating the fine-tuned model')
    print('Loading model from {}'.format(ckpt))
    model = SBertQA(config['model']['pretrained'])
    ckpt = torch.load(os.path.join(ckpt, 'pytorch_model.bin'))
    model.load_state_dict(ckpt)

    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(model=model, args=training_args)

    for i in range(len(config['data']['test_paths'])):
        print('[{}] Evaluating {}'.format(
            i + 1, config['data']['test_paths'][i]))

        tt_set = QADataset3(config['data']['test_paths'][i], 'test')
        logits, _, _ = trainer.predict(tt_set)
        answers = np.argmax(logits, axis=1)

        with open(config['data']['pred_paths'][i], 'w') as fp:
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
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    set_seed(config['train_args']['seed'])

    if args.mode == 'train':
        train(config, val=args.val)
    else:
        eval(config, args.ckpt)
