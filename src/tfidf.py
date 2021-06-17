'''
    File      [ src/tfidf.py ]
    Author    [ Heng-Jui Chang & Chun-Wei Ho (NTUEE) ]
    Synopsis  [ Training for TF-IDF method for the risk task ]
'''

import argparse
import os
import pickle
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from data import ClassificationDataset
from util.word_dict import word_dict

jieba.setLogLevel(20)
for w in word_dict:
    jieba.suggest_freq((w), True)


def remove_numbers(text):
    new_text = []
    for c in text:
        if not c.isdigit():
            new_text.append(c)
    return ''.join(new_text)


def load_data(path, name):
    dataset = ClassificationDataset(
        path, name, val_r=10000, max_orig_sent_len=70)
    data = dataset.data
    X = list(map(lambda x: ' '.join([' '.join(list(jieba.cut(i)))
                                     for i in x[1] if i != '']), data))
    X = [remove_numbers(x) for x in X]
    X = [x.replace('醫師 : ', '').replace('民眾 : ', '') for x in X]
    Y = None if name == 'test' else list(map(lambda x: x[2], data))
    if Y is not None:
        print('class 0 : class 1 = {:.2f} : {:.2f}'
              .format(1. - np.mean(Y), np.mean(Y)))
    return X, Y, dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SVC for Risk Assessment')
    parser.add_argument('--train', type=str, help='Path to training data')
    parser.add_argument('--val', type=str, help='Path to validation data')
    parser.add_argument('--test', type=str, default='',
                        help='Path to testing data')
    parser.add_argument('--ckpt', type=str, default='',
                        help='Path to save ckpt')
    parser.add_argument('--cls', type=str, choices=['svc', 'gdboost'],
                        default='gdboost', help='Classifier type')
    parser.add_argument('--seed', type=int, default=7122, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    X_train, Y_train, _ = load_data(args.train, 'train')
    X_val, Y_val, _ = load_data(args.val, 'train')
    X_test = [] if args.test == '' else load_data(args.test, 'test')[0]

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

    if args.train.find('train_risk_tr-dv') >= 0:
        tfidf.fit(X_train + X_test)
    else:
        tfidf.fit(X_train + X_val + X_test)

    print('Vocab size = {}'.format(len(tfidf.vocabulary_)))
    print('Stop words = {}'.format(len(tfidf.stop_words_)))

    X_train = tfidf.transform(X_train)
    X_val = tfidf.transform(X_val)

    if args.cls == 'svc':
        print('Linear SVC applied.')
        model = LinearSVC(
            penalty='l2',
            loss='squared_hinge',
            dual=True,
            random_state=args.seed
        )
    elif args.cls == 'gdboost':
        print('Gradient Boosting applied.')
        model = GradientBoostingClassifier(
            learning_rate=0.01,
            n_estimators=2000,
            subsample=0.5,
            max_depth=2,
            verbose=0,
            random_state=args.seed
        )

    clf = CalibratedClassifierCV(model, cv=10, n_jobs=2)
    clf.fit(X_train, Y_train)
    Y_train_pred = clf.predict_proba(X_train)[:, 1]
    Y_val_pred = clf.predict_proba(X_val)[:, 1]

    print('=======================')
    print('  Train AUC : {:.5f}'.format(roc_auc_score(Y_train, Y_train_pred)))
    print('  Val   AUC : {:.5f}'.format(roc_auc_score(Y_val, Y_val_pred)))
    print('=======================')

    if args.ckpt != '':
        os.makedirs(args.ckpt, exist_ok=True)
        clf_path = os.path.join(args.ckpt, 'clf.bin')
        vec_path = os.path.join(args.ckpt, 'vec.bin')
        with open(clf_path, 'wb') as fp:
            pickle.dump(clf, fp)
            print('Classification model saved to {}'.format(clf_path))
        with open(vec_path, 'wb') as fp:
            pickle.dump(tfidf, fp)
            print('TfidfVectorizer saved to {}'.format(vec_path))
