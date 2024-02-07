import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

train_data= pd.read_csv("Data/train_set.csv")
test_data=pd.read_csv("Data/test_set.csv")
print("训练集和测试集的维度：", train_data.shape, test_data.shape)

X_train=train_data.drop('FLAG', axis=1)
y_train=train_data['FLAG']
X_test=test_data.drop('FLAG', axis=1)
y_test=test_data['FLAG']
print('划分特征集和标签集后训练集和测试集的维度:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print('训练集和测试集中标签比例：', Counter(y_train), Counter(y_test))

# 模型比较，调用Units_LJR中的函数
from Unit_LJR import model_Comparision_with_mutil_classfiers

model_Comparision_with_mutil_classfiers(X_train, y_train, X_test, y_test)
