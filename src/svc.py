from data import ClassificationDataset
import numpy as np

train_set = ClassificationDataset("data/Train_risk_classification_ans.csv", 'train', val_r=20).data
test_set = ClassificationDataset("data/Train_risk_classification_ans.csv", 'val', val_r=20).data

X_train = list(map(lambda x: ' '.join([' '.join(i) for i in x[1] if i != '']), train_set))
Y_train = list(map(lambda x: x[2], train_set))
X_test = list(map(lambda x: ' '.join([' '.join(i) for i in x[1] if i != '']), test_set))
Y_test = list(map(lambda x: x[2], test_set))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf8', ngram_range=(1,4), stop_words=None, token_pattern=r"(?u)\b\w+\b", max_df=0.6)
tfidf.fit(X_train + X_test)

X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

model = LinearSVC()
clf = CalibratedClassifierCV(model)
clf.fit(X_train, Y_train)
Y_train_pred = clf.predict_proba(X_train)[:,1]
Y_pred = clf.predict_proba(X_test)[:,1]


from sklearn.metrics import roc_auc_score
print('Train AUROC', roc_auc_score(Y_train, Y_train_pred))
print('Test AUROC', roc_auc_score(Y_test, Y_pred))


