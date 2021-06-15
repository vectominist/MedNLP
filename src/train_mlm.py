'''
    File      [ src/train_mlm.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Masked LM training for fine-tuning pre-trained LM ]
'''

import argparse
import yaml
import torch
from transformers import (
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from data import MLMDataset, tokenizer_risk

import logging
logging.disable(logging.WARNING)


def train(config: dict):
    print('Fine-tuning Bert with Medical Dialogues (Masked LM)')
    model = AutoModelForMaskedLM.from_pretrained(
        config['model']['pretrained_model'])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_risk, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=MLMDataset(
            config['data']['train_path'], 'train_all', eda=False),
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
