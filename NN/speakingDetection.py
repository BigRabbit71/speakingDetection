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
from keras.models import load_model

def speakingDetection(X):
    model = Sequential()
    model = load_model('nn_model_7660.h5')

    predY = model.predict_classes(X, batch_size=128)

    return predY


