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

# prepare data

# eg.
# training_data_x = np.random.random((1000, 6))
# training_data_y = np.random.randint(2, size=(1000, 1))
# test_data_x = np.random.random((100,6))
# test_data_y = np.random.randint(2, size=(100, 1))

xData, yData = datasets.load_svmlight_file("training.txt")
# 随机抽取18%的数据集
training_data_x, test_data_x, training_data_y, test_data_y = train_test_split(xData, yData)
print(type(training_data_x.toarray()))

# 将特征值转换成array
training_data_x = training_data_x.toarray()
test_data_x = test_data_x.toarray()

# 将label值从 -1 1 转换为 0 1
training_data_y = (training_data_y[0:]+1)/2
test_data_y = (test_data_y[0:]+1)/2

# Create a model
model = Sequential()

# Input
model.add(Dense(64, activation='relu', input_dim=6))
model.add(Dropout(0.5))
# Hidden
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# Output
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(training_data_x, training_data_y, epochs=3000, batch_size=128)
# Save model
model.save('nn_model.h5')

score = model.evaluate(test_data_x, test_data_y, batch_size=128)
print(score)