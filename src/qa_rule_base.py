'''
    Training for the Multichoice QA
'''

import argparse
import yaml
import os
import csv
import numpy as np

from data import QADatasetRuleBase
from model.qa_model_rulebase import RuleBaseQA

label2answer = ['A', 'B', 'C']


def qa_eval_metrics(eval_pred):
    logits, labels = eval_pred
    if type(logits) == tuple:
        logits = logits[0]
    if type(labels) == tuple:
        labels = labels[0]
    answer = np.argmax(logits, axis=1)
    accuracy = (answer == labels).astype(float).mean()
    return {'acc': accuracy}


def eval(config: dict):
    print('Evaluating the rule-based model')
    model = RuleBaseQA()

    for i in range(len(config['data']['test_paths'])):
        print('[{}] Evaluating {}'.format(
            i + 1, config['data']['test_paths'][i]))

        # tt_set = QADatasetRuleBase(config['data']['test_paths'][i])
        tt_set = QADatasetRuleBase(config['data']['train_path'])
        answers = model.predict(tt_set)
        if 'answer' in tt_set[0]:
            label = np.array([i['answer'] for i in tt_set])
            print("Accuracy: %s" % (label == answers).mean())

        os.makedirs(os.path.split(config['data']['pred_paths'][i])[0], exist_ok=True)
        with open(config['data']['pred_paths'][i], 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'answer'])
            ids = tt_set.get_ids()
            assert len(ids) == len(answers)
            for j in range(len(answers)):
                writer.writerow([str(ids[j]), label2answer[answers[j]]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training for QA Evaluation')
    parser.add_argument('--config', type=str, help='Path to config')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    eval(config)
