import pandas as pd
import io
from sklearn.metrics import confusion_matrix,  accuracy_score, roc_auc_score, precision_score, recall_score, mean_squared_error, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# df = pd.read_csv(io.StringIO('titanic.csv'))
# print(df.info)
from sklearn.svm import SVC

import csv
data = pd.read_csv('winequality-red.csv')
# print(df.columns)
# target = df['quality']
# df.drop(['quality'], axis=1, inplace=True)
# print(df.shape)
# print(target.shape)
# # with open('winequality-red.csv', newline='', encoding='utf-8') as f:
# #     reader = csv.reader(f)
# #     for row in reader:
# #         print(row)

print(data['quality'].value_counts())

bins = (2, 6.5, 8)

group_names = [0, 1] # 0:Bad, 1:Good

data['quality'] = pd.cut(data['quality'], bins=bins, labels=group_names)

print(data['quality'].value_counts())
"""
5    681
6    638
7    199
4     53
8     18
3     10
Name: quality, dtype: int64
0    1382
1     217
"""

X = data.drop('quality', axis=1)

y = data['quality']



X_train, X_test, y_train, y_test = X[:1300], X[1300:], y[:1300], y[1300:]
param_grid = [

    {'C': [0.1, 1, 10], 'degree':[2, 3], 'kernel': ['poly'], 'random_state': [1234]},

    {'C': [0.1, 1, 10], 'gamma': [0.0001, 0.001, 0.01],

     'kernel': ['linear', 'rbf'],
     'random_state': [1234]}

]
bestSVC = SVC()
gsSVC = GridSearchCV(bestSVC, param_grid, n_jobs=4)
gsSVC.fit(X_train, y_train)
y_test_pred = gsSVC.predict(X_test)

ConMatrix = confusion_matrix(y_test, y_test_pred)
ClassReport = classification_report(y_test, y_test_pred)
ROC = roc_auc_score(y_test, y_test_pred)
print("Confusion Matrix Score:", ConMatrix)
print("Classification Report: ", ClassReport)
print("ROC AUC Score: ", ROC)
"""
Confusion Matrix Score: 
[[275   0]
 [ 24   0]]
Classification Report:
                precision    recall  f1-score   support

           0       0.92      1.00      0.96       275
           1       0.00      0.00      0.00        24

    accuracy                           0.92       299
   macro avg       0.46      0.50      0.48       299
weighted avg       0.85      0.92      0.88       299

ROC AUC Score:  0.5
"""