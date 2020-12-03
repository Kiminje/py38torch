"""위의 mnist데이터를 가지고 렌덤포리스트의 분류기를 만들어보고 , 엑스트라 트리
분류기 등 여러가지 분류기를 훈련시키고 검증 세트에서 개개의 분류기보다 더 높
은 성능을 내도록 직접, 간접 투표방법을 사용해 앙상블로 연결해보고 개개의 분류
기와 비교해서 djEjgr 성능이 향상되는지를 관찰하시오"""

from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import pydot
# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,  accuracy_score, roc_auc_score, precision_score, recall_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import matplotlib
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
##boosting xgboost, gbm implementation
from sklearn.model_selection import train_test_split

X,y = mnist['data'], mnist['target']
print(X.shape) #    (70000, 784)
print(y.shape) #    (70000, )
# print(mnist.shape)
# print(mnist)
import matplotlib as mlt
import matplotlib.pyplot as plt
# some_digit=X[0]
# some_digit_image=some_digit.reshape(28,28)
# plt.imshow(some_digit_image, cmap="binary")
# plt.show()
y[0]#문자형이므로 숫자형으로 바꾸어야 함
# y = to_categorical(y)
y=y.astype(np.uint8)
X = X.astype(np.float)
X /= 255
# import sklearn
# y = sklearn.preprocessing.OneHotEncoder(y)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# ds=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
#                        max_depth=10, max_features=None, max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=200,
#                        min_weight_fraction_leaf=0.0, presort='deprecated',
#                        random_state=156, splitter='best')
# print("ds end")
# ds.fit(X_train, y_train)
# print(mnist["class"])
# Class = ["{}".format(i for i in range(0,10))]
# print(Class)
# print(mnist.shape)
# print(ds.shape)
#
# from sklearn.tree import export_graphviz
# with open("mnist.dot", "w") as f:
#     f = export_graphviz(ds,
#                         out_file = f,
#                         feature_names = mnist.feature_names,
#                         class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
#                         filled = True,
#                         rounded = True,
#                        impurity=True)
# import graphviz
# with open("mnist.dot") as f:
#          dot_graph=f.read()
# graphviz.Source(dot_graph)
# (graph,) = pydot.graph_from_dot_file('mnist.dot')
# graph.write_png('mnist.png')
# y_pred = ds.predict(X_test)
# confusion = confusion_matrix( y_test, y_pred)
# accuracy = accuracy_score(y_test , y_pred)
# print('오차행렬\n', confusion)
# print('정확도: {0:.4f}'.format(accuracy ))
# with open("mnist_pred.dot", "w") as f:
#     f = export_graphviz(ds,
#                         out_file = f,
#                         feature_names = mnist.feature_names,
#                         class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
#                         filled = True,
#                         rounded = True,
#                        impurity=True)
#
# with open("mnist_pred.dot") as f:
#     dot_graph = f.read()
# graphviz.Source(dot_graph)
# (graph,) = pydot.graph_from_dot_file('mnist.dot')
# graph.write_png('mnist_pred.png')
#
# ds3=RandomForestClassifier(random_state=156).fit(X_train, y_train)
#
# pred=ds3.predict(X_test)
#
# rsquared_train = ds3.score(X_train, y_train)
# rsquared_test = ds3.score(X_test, y_test)
# confusion = confusion_matrix( y_test, pred)
# accuracy = accuracy_score(y_test , pred)
#
# print('오차 행렬')
# print(confusion)
# print('정확도: {0:.4f}'.format(accuracy ))
# sns.barplot(x=ds3.feature_importances_,y=mnist.feature_names)

import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


X_train= pd.DataFrame(X_train, columns=mnist.feature_names)
y_train=pd.DataFrame(y_train)
y_train = y_train.values.ravel()
# rf_w = RandomForestClassifier(random_state=42, class_weight = 'balanced')
# rf_w.fit(X_train, y_train.values.ravel())
#
# pred=rf_w.predict(X_test)
# print(rf_w.__class__.__name__, accuracy_score(y_test, pred))
import sklearn

#
# sns.barplot(x=rf_w.feature_importances_, y=mnist.feature_names)
# plt.show()
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# in gridsearch, only best model is suggested
#   method for choosing best hyperparameters
# param = {
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'None', 'log2'],
#     'max_depth': [4, 5, 6, 7, 8],
#     'criterion':['gini', 'entropy']
# }
#
# print(y_train.shape)
# grid_rf_w = GridSearchCV(estimator=rf_w, param_grid=param, cv=3)
# grid_rf_w.fit(X_train, y_train.values.ravel())
# print(grid_rf_w.best_params_)
# print(grid_rf_w.best_score_)
#
# log_clf = LogisticRegression(solver="lbfgs", random_state=42)
# rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
#
# #Hard Voting
# voting_clf = VotingClassifier(
#     estimators=[('lr', log_clf), ('rf', rnd_clf)],
#     voting='hard')
# voting_clf.fit(X_train, y_train)
# voting_pred=voting_clf.predict(X_test)

from sklearn.metrics import accuracy_score
#
# for clf in (log_clf, rnd_clf, voting_clf):
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
#
#
#
# # Soft Voting
# log_clf = LogisticRegression(solver="lbfgs", random_state=42)
# rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
#
#
# voting_clf = VotingClassifier(
#     estimators=[('lr', log_clf), ('rf', rnd_clf)],
#     voting='soft')
#
# voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
#
# for clf in (log_clf, rnd_clf, voting_clf):
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
#
# print("Adaboosting start")
# start_time = time.time()
# ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200, algorithm='SAMME.R',learning_rate=0.5, random_state=42)
# ada_clf.fit(X_train, y_train)
# ada_pred = ada_clf.predict(X_test)
# accuracy = accuracy_score(y_test, ada_pred)
# print("정확도: {0:.4f}".format( accuracy))
# print("수행시간: {0:.1f}".format(time.time()-start_time))

from matplotlib.colors import ListedColormap
#
# def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.5, contour=True):
#     x1s = np.linspace(axes[0], axes[1], 100)
#     x2s = np.linspace(axes[2], axes[3], 100)
#     x1, x2 = np.meshgrid(x1s, x2s)
#     X_new = np.c_[x1.ravel(), x2.ravel()]
#     y_pred = clf.predict(X_new).reshape(x1.shape)
#     custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
#     plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
#     if contour:
#         custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
#         plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
#     plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
#     plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
#     plt.axis(axes)
#     plt.xlabel(r"$x_1$", fontsize=18)
#     plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
#     plt.show()
#
# plot_decision_boundary(ada_clf, X, y)

#Gradient Boosting
# print("GBM start!")
from sklearn.ensemble import GradientBoostingClassifier
# start_time = time.time()
# gb_clf = GradientBoostingClassifier(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
# gb_clf.fit(X_train, y_train)
# gb_pred = gb_clf.predict(X_test)
# accuracy = accuracy_score(y_test, gb_pred)
# print("정확도: {0:.4f}".format( accuracy))
# print("수행시간: {0:.1f}".format(time.time()-start_time))
#
# #Staged Method
# errors=[mean_squared_error(y_test,y_pred)
#   for y_pred in gb_clf.staged_predict(X_test) ]
# print("errors", errors)
# bst_n_estimators=np.argmin(errors)+1
# print("optimized number of estimator : ", bst_n_estimators)
# gbrt_best =GradientBoostingClassifier(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
# gbrt_best.fit(X_train, y_train)
# gbrt_pred=gbrt_best.predict(X_test)
# accuracy  = accuracy_score(y_test, gbrt_pred)
# print("정확도: {0:.4f}:".format( accuracy))
#
# # Early Stopping
# gbwt =GradientBoostingClassifier(max_depth=2,warm_start=True)
# min_val_error = float("inf")
# error_going_up=0
# for n_estimators in range(1,120):
#   gbwt.n_estimators=n_estimators
#   gbwt.fit(X_train, y_train)
#   y_pred = gbwt.predict(X_test)
#   val_error = mean_squared_error(y_test, y_pred)
#   if val_error < min_val_error:
#     min_val_error = val_error
#     error_going_up=0
#   else:
#     error_going_up +=1
#     if error_going_up==5:
#       break
# print(min_val_error)

## XGBoost

import xgboost
# xgb_reg = xgboost.XGBClassifier()
# xgb_reg.fit (X_train, y_train)
# y_pred = xgb_reg.predict(X_test)
#
# xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=2)
# y_pred = xgb_reg.predict(X_test)
"""models = [
    ('ada', AdaBoostClassifier()),
    ('bc', BaggingClassifier()),
    ('etc',ExtraTreesClassifier()),
    ('gbc', GradientBoostingClassifier()),
    ('rfc', RandomForestClassifier()),
    ('knn', KNeighborsClassifier()),
    ('svc', SVC(probability=True)),
    ('xgb', XGBClassifier()),
    ('lgbm', LGBMClassifier()),
    ('dtc', DecisionTreeClassifier()),
    ('lr', LogisticRegressionCV()),
    ('ridge', RidgeClassifier()),
]"""
"""
XGBoost : xgb_reg ''
Gradient Boosting
Adaboosting
random forest
train 4 of models using gride search and apply on soft & hard voting
"""
with open("mnist_result.csv", "w", encoding="UTF-8") as f:
    f.write("Classifier,Accuracy,parameter list")
    k_fold = StratifiedKFold(n_splits=3 ,shuffle=True, random_state=42)
    RF_best = RandomForestClassifier()
    RF_param = {"max_depth": [10],
                'max_features': ['auto'],
                'n_estimators': [500],
                'criterion':['entropy'],
                'random_state': [42]}
    gsRF = GridSearchCV(RF_best, RF_param, cv=k_fold, scoring='accuracy', verbose=1,n_jobs=4)
    gsRF.fit(X_train, y_train)
    RF_best = gsRF.best_estimator_
    print(gsRF.__class__, gsRF.best_params_, gsRF.best_score_)
    RF_pred = RF_best.predict(X_test)
    accuracy = accuracy_score(y_test, RF_pred)
    """
    <class 'sklearn.model_selection._search.GridSearchCV'> {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'auto', 'n_estimators': 500, 'random_state': 42} 0.9489333333333333
    RF 정확도: 0.9527
"""
    print("RF 정확도: {0:.4f}".format(accuracy))
    f.write("{}".format(gsRF.__class__) + ',' + "{0:.4f}".format(accuracy) + ',' + "{}".format(gsRF.best_params_) + '\n')

    Ada_best = AdaBoostClassifier()
    Ada_param = {'n_estimators': [500],
                 'base_estimator': [DecisionTreeClassifier(max_depth=2)],
                 'learning_rate': [0.1, 0.05],
                'algorithm': ['SAMME.R'],
                 'random_state': [42]}
    gsAda = GridSearchCV(Ada_best, Ada_param, cv=k_fold, scoring='accuracy', verbose=1,n_jobs=4)
    gsAda.fit(X_train, y_train)
    Ada_best = gsAda.best_estimator_
    print(gsAda.classes_, gsAda.best_params_, gsAda.best_score_)
    Ada_pred = Ada_best.predict(X_test)
    accuracy = accuracy_score(y_test, Ada_pred)
    print("Ada 정확도: {0:.4f}".format(accuracy))
    f.write("{}".format(gsAda.__class__) + ',' + "{0:.4f}".format(accuracy) + ',' + "{}".format(gsAda.best_params_) + '\n')

    GBM_best = GradientBoostingClassifier()
    GBM_param = {"max_depth": [4],
                'max_features': ['auto'],
                'n_estimators': [200, 500],
                 'learning_rate': [0.1],
                 'random_state': [42]}
    gsGBM = GridSearchCV(GBM_best, GBM_param, cv=k_fold, scoring='accuracy', verbose=1, n_jobs=4)
    gsGBM.fit(X_train, y_train)
    GBM_best = gsGBM.best_estimator_
    print(gsGBM.classes_, gsGBM.best_params_, gsGBM.best_score_)
    GBM_pred = GBM_best.predict(X_test)
    accuracy = accuracy_score(y_test, GBM_pred)
    print("GBM 정확도: {0:.4f}".format(accuracy))
    sent = "{}".format(gsGBM.__class__) + ',' + "{0:.4f}".format(accuracy) + ',' + "{}".format(gsGBM.best_params_) + '\n'
    f.write(sent)


    XGB_best = xgboost.XGBClassifier()
    XG_param = {"max_depth": [8],
                'min_child_weight': [3,6],
                'gamma': [0, 1e-1],
                 'learning_rate': [0.5],
                'random_state': [42]}
    gsXG = GridSearchCV(XGB_best, XG_param, cv=k_fold, scoring='accuracy', verbose=1,n_jobs=4)
    gsXG.fit(X_train, y_train)
    XG_best = gsXG.best_estimator_
    print(gsXG.classes_, gsXG.best_params_, gsXG.best_score_)
    XG_pred = XG_best.predict(X_test)
    accuracy = accuracy_score(y_test, XG_pred)
    print("XGB 정확도: {0:.4f}".format(accuracy))
    f.write("{}".format(gsXG.__class__) + ',' + "{0:.4f}".format(accuracy) + ',' + "{}".format(gsXG.best_params_) + '\n')


    # ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200, algorithm='SAMME.R',learning_rate=0.5, random_state=42)
    voting_clf = VotingClassifier(
        estimators=[('rfc', RF_best), ('xgb', XG_best), ('gbc', GBM_best), ('ada', Ada_best)],
        voting='hard', verbose=1, n_jobs=4)
    voting_clf.fit(X_train, y_train)
    voting_pred=voting_clf.predict(X_test)
    accuracy  = accuracy_score(y_test, voting_pred)
    print("hard voting 정확도: {0:.4f}".format(accuracy))
    f.write("hard voting"+ ',' + "{0:.4f}".format(accuracy) + ',' + "{}".format(voting_clf.best_params_) + '\n')

    voting_clf1 = VotingClassifier(
        estimators=[('rfc', RF_best), ('xgb', XG_best), ('gbc', GBM_best), ('ada', Ada_best)],
        voting='soft', verbose=1, n_jobs=4)
    voting_clf1.fit(X_train, y_train)
    voting_pred=voting_clf1.predict(X_test)
    accuracy  = accuracy_score(y_test, voting_pred)
    print("soft voting 정확도: {0:.4f}".format(accuracy))
    f.write("soft voting" + ',' + "{0:.4f}".format(accuracy) + ',' + "{}".format(voting_clf1.best_params_) + '\n')
