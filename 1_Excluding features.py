import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, KFold
from imblearn import over_sampling,under_sampling,combine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data= pd.read_csv("Data/ETH_Transaction_REC20_Data.csv")
print("ETH_Transaction_REC20_Data的基本信息：")
print(data.head(10), '\n', data.shape)
print(Counter(data['FLAG']))   # 7662:2179

data.columns = data.columns.str.strip()  # 把字符串头和尾的空格，以及位于头尾的\n \t之类给删掉
data.columns = data.columns.str.replace(' ', '_')  # 将字符串中的空格用_代替

# 存储数据集的数值型连续特征的 统计信息
print(data.describe())
pd.DataFrame(data.describe().T).to_excel('Data/data_1.describe.xlsx', index=True)

# 查看数据信息
print(data.info())
print(data.nunique())

# 从数据信息可看出，有7个ERC20特征的样本值全为0，这对建模无用，因此剔除，同时剔除index、Address、Unnamed:_0特征
data_cols = [col for col in data.columns if col not in ['Unnamed:_0',
                                                        'Index',
                                                        'Address',
                                                        'ERC20_avg_time_between_sent_tnx',
                                                        'ERC20_avg_time_between_rec_tnx',
                                                        'ERC20_avg_time_between_rec_2_tnx',
                                                        'ERC20_avg_time_between_contract_tnx',
                                                        'ERC20_min_val_sent_contract',
                                                        'ERC20_max_val_sent_contract',
                                                        'ERC20_avg_val_sent_contract'
                                                        ]]

# 存储特征名称
data_name = data.columns.values.tolist()
print(data_name)
data_1=data[data_cols]
print(data_1.shape)


# 保存数据集
# data_1.to_csv(path_or_buf=r'Data/Data_1.csv', index=None)