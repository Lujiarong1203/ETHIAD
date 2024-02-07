import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data= pd.read_csv("Data/Data_2.csv")
print(data.shape)
print(data.isnull().sum())

# 查看字符串特征
index=data.ERC20_most_sent_token_type
print(index.value_counts())

data_dummies = pd.get_dummies(data, columns=['ERC20_most_sent_token_type', 'ERC20_most_rec_token_type'], dummy_na=True)
print(data_dummies.head(5), '\n', data.shape)

# 读取自模块
from Unit_LJR import model_comparison_with_not_split
RF=RandomForestClassifier()
XG=XGBClassifier()
# 读取原数据，去除2个字符串特征
data_not_dummies=data.drop(['ERC20_most_sent_token_type', 'ERC20_most_rec_token_type'], axis=1)

# 比较One-Hot编码前后模型性能
model_comparison_with_not_split(data_not_dummies, XG, random_seed=1234, name='data_not_dummies')
model_comparison_with_not_split(data_dummies, XG, random_seed=1234, name='data_get_dummies')

# 结果显示，One-Hot编码后模型性能提升，因此保存编码后的数据集
data_dummies.to_csv(path_or_buf=r'Data/Data_3.csv', index=None)