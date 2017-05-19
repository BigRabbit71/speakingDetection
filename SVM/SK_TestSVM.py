# 导入数值计算库
import numpy as np
# 导入科学计算库
import pandas as pd

from sklearn import datasets, linear_model
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from speakingDetection import *

xData, yData = datasets.load_svmlight_file("training.txt")

# 随机抽取的数据集
training_data_x, test_data_x, training_data_y, test_data_y = train_test_split(xData, yData)
real_result = test_data_y

prediction = speakingDetection(test_data_x)

report = classification_report(test_data_y, prediction)
print(report)
