'''
    File      [ src/tfidf_test.py ]
    Author    [ Heng-Jui Chang & Chun-Wei Ho (NTUEE) ]
    Synopsis  [ Testing for TF-IDF method for the risk task ]
'''

import argparse
import csv
import os
import pickle
import numpy as np
import jieba

from tfidf import load_data
from util.word_dict import word_dict

import logging
logging.disable(logging.WARNING)

for w in word_dict:
    jieba.suggest_freq((w), True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SVC for Risk Assessment')
    parser.add_argument('--test', type=str, help='Path to testing data')
    parser.add_argument('--ckpt', type=str, help='Path to load ckpt')
    parser.add_argument('--out', type=str, default='',
                        help='Path to output file')
    parser.add_argument('--seed', type=int, default=7122, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    X_test, _, tt_set = load_data(args.test, 'test')

    clf_path = os.path.join(args.ckpt, 'clf.bin')
    vec_path = os.path.join(args.ckpt, 'vec.bin')
    with open(vec_path, 'rb') as fp:
        tfidf = pickle.load(fp)
        print('Loaded TfidfVectorizer from {}'.format(vec_path))
    with open(clf_path, 'rb') as fp:
        clf = pickle.load(fp)
        print('Loaded classification model from {}'.format(clf_path))

    print('Vocab size = {}'.format(len(tfidf.vocabulary_)))
    print('Stop words = {}'.format(len(tfidf.stop_words_)))

    X_test = tfidf.transform(X_test)
    Y_test_pred = clf.predict_proba(X_test)[:, 1]

    if args.out != '':
        with open(args.out, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['article_id', 'probability'])
            ids = tt_set.get_ids()
            assert len(ids) == len(Y_test_pred)
            for i in range(len(Y_test_pred)):
                writer.writerow([str(ids[i]), Y_test_pred[i]])
            print('Results saved to {}'.format(args.out))
