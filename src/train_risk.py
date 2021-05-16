'''
Training for the Risk Evaluation Task
'''

import argparse
import yaml

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)

from data import ClassificationDataset


def train(config: dict):
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['pretrained'], num_labels=2)
    training_args = TrainingArguments(**config['train_args'])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ClassificationDataset(config['data']['path'], 'train'),
        eval_dataset=ClassificationDataset(config['data']['path'], 'val'),
    )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training for Risk Evaluation')
    parser.add_argument('--name', type=str, help='Name for model')
    parser.add_argument('--config', type=str, help='Path to config')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    set_seed(config['train_args']['seed'])
    train(config)
