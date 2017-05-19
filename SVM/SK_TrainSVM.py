# 导入数值计算库
import numpy as np
# 导入科学计算库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from sklearn.externals import joblib

xData, yData = datasets.load_svmlight_file("training.txt")
# print(yData[:30])

# 随机抽取数据集
training_data_x, test_data_x, training_data_y, test_data_y = train_test_split(xData, yData)
# print(len(training_data_y))

# Support Vector Classifier
clf = SVC()

# 网格搜索
C = np.logspace(-1,1,5,base=2)
gamma = np.logspace(-2,5,5,base=2)
param_grid = dict(C=C, gamma=gamma)
grid = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1)
# grid_data_x = training_data_x[10000:14000]
# grid_data_y = training_data_y[10000:14000]
# grid_result = grid.fit(grid_data_x, grid_data_y)
grid_result = grid.fit(training_data_x, training_data_y)

print(grid.best_score_)
print(grid.best_estimator_)
bestC = grid.best_estimator_.C
bestGamma = grid.best_estimator_.gamma



# Train & Test
clf.fit(training_data_x, training_data_y)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape=None, degree=3, gamma=9.5136569200217682, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=True)
SVC(C=bestC, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma=bestGamma, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=True)

# joblib.dump(clf, 'svcmodel.pkl')

real_result = test_data_y
prediction = clf.predict(test_data_x)

report = classification_report(test_data_y, prediction)
print(report)
print(clf.score(test_data_x, test_data_y))

