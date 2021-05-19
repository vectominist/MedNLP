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
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from data import MLMDataset, tokenizer_risk


def train(config: dict):
    print('Fine-tuning Bert with Medical Dialogues (Masked LM)')
    model = AutoModelForMaskedLM.from_pretrained('ckiplab/albert-tiny-chinese')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_risk, mlm=True, mlm_probability=0.15)
    
    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=MLMDataset(config['data']['train_path'], 'train_all'),
        data_collator=data_collator
    )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MLM Fine-tuning')
    parser.add_argument('--config', type=str, help='Path to config')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    set_seed(config['train_args']['seed'])
    train(config)
