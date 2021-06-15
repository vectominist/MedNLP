'''
    File      [ src/gdboost_grid_search.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Grid search for gradient boosting classifier ]
'''

import argparse
import csv
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from data import ClassificationDataset
from util.word_dict import word_dict
from tfidf import load_data

import logging
logging.disable(logging.WARNING)

for w in word_dict:
    jieba.suggest_freq((w), True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Grid search for GDBoosting')
    parser.add_argument('--train', type=str, help='Path to training data')
    parser.add_argument('--val', type=str, help='Path to validation data')
    parser.add_argument('--seed', type=int, default=7122, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    X_train, Y_train, _ = load_data(args.train, 'train')
    X_val, Y_val, _ = load_data(args.val, 'train')

    tfidf = TfidfVectorizer(
        encoding='utf8',
        analyzer='word',  # word
        stop_words=None,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 1),  # (1, 1)
        max_df=0.9,  # 0.9
        min_df=3,  # 3
        max_features=None,
        norm='l2',
        sublinear_tf=True
    )

    tfidf.fit(X_train + X_val)

    print('Vocab size = {}'.format(len(tfidf.vocabulary_)))
    print('Stop words = {}'.format(len(tfidf.stop_words_)))

    X_train = tfidf.transform(X_train)
    X_val = tfidf.transform(X_val)

    # lr_list = [0.2, 0.1, 0.05, 0.02, 0.01]
    # n_est_list = [50, 60, 70, 80, 90, 100, 120,
    #               140, 160, 180, 200, 250, 300, 350, 400, 500]
    # lr_list = [0.05, 0.02, 0.01]
    # n_est_list = [100, 200, 300, 400, 500, 600, 700,
    #               800, 900, 1000, 1200, 1500, 2000]
    lr_list = [0.01]
    n_est_list = [2000]
    subsample_list = [0.5]

    best_lr, best_n_est, best_sub = 0., 0, 0
    best_auc = 0.

    for lr in lr_list:
        for n_est in n_est_list:
            for sub in subsample_list:
                model = GradientBoostingClassifier(
                    learning_rate=lr,
                    n_estimators=n_est,
                    subsample=sub,
                    max_depth=2,
                    verbose=0,
                    random_state=args.seed
                )

                clf = CalibratedClassifierCV(model, cv=10, n_jobs=2)
                clf.fit(X_train, Y_train)
                Y_val_pred = clf.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(Y_val, Y_val_pred)

                if val_auc > best_auc:
                    best_lr, best_n_est, best_sub = lr, n_est, sub
                    best_auc = val_auc
                print('lr = {} , n_est = {:4d} , sub = {} , AUC = {:.5f}'
                      .format(lr, n_est, sub, val_auc))

    print()
    print('Best lr    = {}'.format(best_lr))
    print('Best n_est = {}'.format(best_n_est))
    print('Best sub   = {}'.format(best_sub))
    print('Best auc   = {}'.format(best_auc))
