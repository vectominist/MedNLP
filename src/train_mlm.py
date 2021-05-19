'''
Masked LM Fine-tuning
'''

import argparse
import yaml
import os
import csv
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    set_seed
)
from data import MLMDataset


def train(config: dict):
    print('Fine-tuning Bert with Medical Dialogues (Masked LM)')
    model = AutoModelForMaskedLM.from_pretrained('ckiplab/bert-base-chinese')
    
    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=MLMDataset(config['data']['train_path'], 'train'),
        eval_dataset=MLMDataset(config['data']['train_path'], 'val')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MLM Fine-tuning')
    parser.add_argument('--config', type=str, help='Path to config')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    set_seed(config['train_args']['seed'])
    train(config)
