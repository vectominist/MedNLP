'''
    Training for the Multichoice QA
'''

import argparse
import yaml
import os
import csv
import numpy as np

from data import QADatasetRuleBase
from model import RuleBaseQA, RuleBaseQA2

label2answer = ['A', 'B', 'C']


def eval(config: dict):
    print('Evaluating the rule-based model')
    # model = RuleBaseQA()
    model = RuleBaseQA2()

    for i in range(len(config['data']['test_paths'])):
        print('[{}] Evaluating {}'.format(
            i + 1, config['data']['test_paths'][i]))

        tt_set = QADatasetRuleBase(config['data']['test_paths'][i])
        answers, is_inv = model.predict(tt_set)

        if tt_set[0]['answer'] is not None:
            label = np.array([i['answer'] for i in tt_set])
            answers_nor = answers[is_inv != True]
            answers_inv = answers[is_inv == True]
            label_nor = np.array(
                [i['answer'] for idx, i in enumerate(tt_set) if not is_inv[idx]])
            label_inv = np.array(
                [i['answer'] for idx, i in enumerate(tt_set) if is_inv[idx]])
            print('All ACC     = {:.4f}'.format((label == answers).mean()))
            print('Normal ACC  = {:.4f}  ({} samples)'
                  .format((label_nor == answers_nor).mean(), len(answers_nor)))
            print('Inverse ACC = {:.4f}  ({} samples)'
                  .format((label_inv == answers_inv).mean(), len(answers_inv)))

        if config['data']['pred_paths'][i] == '':
            continue
        os.makedirs(os.path.split(
            config['data']['pred_paths'][i])[0], exist_ok=True)
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
