# 导入数值计算库
import numpy as np
# 导入科学计算库
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import keras
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

from speakingDetection import *
from sklearn.metrics import classification_report


# prepare data
xData, yData = datasets.load_svmlight_file("training.txt")
# 随机抽取18%的数据集
training_data_x, test_data_x, training_data_y, test_data_y = train_test_split(xData, yData)
# print(type(training_data_x.toarray()))

training_data_x = training_data_x.toarray()
training_data_y = np.array(training_data_y, dtype=int)

test_data_x = test_data_x.toarray()
test_data_y = np.array(test_data_y, dtype=int)

num_train = len(training_data_y)
num_test = len(test_data_y)

training_data_y = training_data_y.reshape(num_train, 1)
test_data_y = test_data_y.reshape(num_test, 1)
# 将label值转换为 0 或 1
training_data_y = (training_data_y[0:]+1)/2
test_data_y = (test_data_y[0:]+1)/2



real_result = test_data_y
prediction = speakingDetection(test_data_x)

report = classification_report(test_data_y, prediction)
print(report)