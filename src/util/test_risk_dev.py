import argparse
import csv
import numpy as np
from sklearn.metrics import roc_auc_score


def read_dev_csv(path):
    with open(path, 'r') as fp:
        rows = csv.reader(fp)
        data = []
        for i, r in enumerate(rows):
            if i == 0:
                continue
            data.append(int(r[3]))
        return np.array(data)


def read_result_csv(path):
    with open(path, 'r') as fp:
        rows = csv.reader(fp)
        data = []
        for i, r in enumerate(rows):
            if i == 0:
                continue
            data.append(float(r[1]))
        return np.array(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute AUC on dev set.')
    parser.add_argument('--dev', type=str, help='Path to dev .csv')
    parser.add_argument('--res', type=str, help='Path to result .csv')
    args = parser.parse_args()

    labels = read_dev_csv(args.dev)
    scores = read_result_csv(args.res)
    auc = roc_auc_score(labels, scores)

    print('AUC = {:.5f}'.format(auc))
