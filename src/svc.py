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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(X_train, Y_train)
Y_train_pred = model.predict(X_train)
Y_pred = model.predict(X_test)

print('Train acc', (Y_train_pred == Y_train).mean())
print('Test acc', (Y_pred == Y_test).mean())




