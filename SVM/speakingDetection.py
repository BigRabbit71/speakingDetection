# 导入数值计算库
import numpy as np
# 导入科学计算库
import pandas as pd

from sklearn import datasets, linear_model
from sklearn.svm import SVC
from sklearn.externals import joblib


def speakingDetection(X):
    clf = SVC()

    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    # decision_function_shape=None, degree=3, gamma=9.5136569200217682, kernel='rbf',
    # max_iter=-1, probability=False, random_state=None, shrinking=True,
    # tol=0.001, verbose=True)

    clf = joblib.load('svcmodel.pkl')

    predY = clf.predict(X)
    return predY


