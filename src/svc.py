import argparse
import csv
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from data import ClassificationDataset
from util.word_dict import word_dict

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
    # X = list(map(lambda x: ' '.join([' '.join(i)
    #                                  for i in x[1] if i != '']), data))
    # X = list(map(lambda x: ' '.join([''.join(i)
    #                                  for i in x[1] if i != '']), data))
    Y = None if name == 'test' else list(map(lambda x: x[2], data))
    if Y is not None:
        print('class 0 : class 1 = {:.2f} : {:.2f}'
              .format(1. - np.mean(Y), np.mean(Y)))
    return X, Y, dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SVC for Risk Assessment')
    parser.add_argument('--train', type=str, help='Path to training data')
    parser.add_argument('--val', type=str, help='Path to validation data')
    parser.add_argument('--test', type=str, help='Path to testing data')
    parser.add_argument('--out', type=str, help='Path to output file')
    args = parser.parse_args()

    X_train, Y_train, _ = load_data(args.train, 'train')
    X_val, Y_val, _ = load_data(args.val, 'train')
    X_test, _, tt_set = load_data(args.test, 'test')

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

    if args.val != args.test:
        if args.train.find('train_risk_tr-dv') >= 0:
            tfidf.fit(X_train)
        else:
            tfidf.fit(X_train + X_val + X_test)
    else:
        tfidf.fit(X_train + X_val)

    # print(tfidf.vocabulary_)
    print('Vocab size = {}'.format(len(tfidf.vocabulary_)))
    print('Stop words = {}'.format(len(tfidf.stop_words_)))
    # print(tfidf.stop_words_)

    X_train = tfidf.transform(X_train)
    X_val = tfidf.transform(X_val)
    X_test = tfidf.transform(X_test)

    # model = LinearSVC(
    #     penalty='l2',
    #     loss='squared_hinge',
    #     dual=True,
    #     random_state=7122
    # )
    # model = DecisionTreeClassifier(
    #     splitter='best',
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_features=None,
    #     random_state=7122
    # )
    # model = AdaBoostClassifier(
    #     n_estimators=40,
    #     learning_rate=1.,
    #     algorithm='SAMME.R',
    #     random_state=7122
    # )
    model = GradientBoostingClassifier(
        learning_rate=0.1,
        n_estimators=80,
        subsample=0.8,
        max_depth=2,
        verbose=0,
        random_state=7122
    )

    clf = CalibratedClassifierCV(
        model,
        cv=10
    )
    clf.fit(X_train, Y_train)
    Y_train_pred = clf.predict_proba(X_train)[:, 1]
    Y_val_pred = clf.predict_proba(X_val)[:, 1]
    Y_test_pred = clf.predict_proba(X_test)[:, 1]

    print('Train AUROC', roc_auc_score(Y_train, Y_train_pred))
    print('Val AUROC', roc_auc_score(Y_val, Y_val_pred))

    with open(args.out, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['article_id', 'probability'])
        ids = tt_set.get_ids()
        assert len(ids) == len(Y_test_pred)
        for i in range(len(Y_test_pred)):
            writer.writerow([str(ids[i]), Y_test_pred[i]])
