import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.impute import KNNImputer
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn import over_sampling,under_sampling,combine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data= pd.read_csv("Data/Data_1.csv")
print(data.shape)

# 查看缺失值情况
print(data.isnull().sum())

na_ratio=data.isnull().sum()[data.isnull().sum()>=0].sort_values(ascending=False)/len(data)
na_sum=data.isnull().sum().sort_values(ascending=False)
print(na_ratio)   # 18个特征存在缺失值，都是ERC20特征

# 绘制缺失值矩阵图
miss_fea_col=['Avg_min_between_sent_tnx', 'Avg_min_between_received_tnx', 'Sent_tnx', 'Received_Tnx', 'Number_of_Created_Contracts',
              'Unique_Received_From_Addresses', 'Unique_Sent_To_Addresses', 'min_value_received', 'max_value_received',
              'avg_val_received', 'min_val_sent', 'Total_ERC20_tnxs', 'ERC20_total_Ether_received',
              'ERC20_total_ether_sent', 'ERC20_most_sent_token_type', 'ERC20_most_rec_token_type']
data_miss=data[miss_fea_col]
missng1=msno.matrix(data_miss,labels=True,label_rotation=20,fontsize=20,figsize=(15,8))#绘制缺失值矩阵图
plt.savefig('Fig.2(a).jpg', bbox_inches='tight',pad_inches=0,dpi=1500,)
plt.show()


# 选择KNN算法填充缺失值，因为KNN算法只能填充数值型特征，因此，选择16个数值型特征进行填充
features_null_col=['Total_ERC20_tnxs', 'ERC20_total_Ether_received', 'ERC20_total_ether_sent', 'ERC20_total_Ether_sent_contract',
                   'ERC20_uniq_sent_addr', 'ERC20_uniq_rec_addr', 'ERC20_uniq_sent_addr.1', 'ERC20_uniq_rec_contract_addr',
                   'ERC20_min_val_rec', 'ERC20_max_val_rec', 'ERC20_avg_val_rec', 'ERC20_min_val_sent',
                   'ERC20_max_val_sent', 'ERC20_avg_val_sent', 'ERC20_uniq_sent_token_name', 'ERC20_uniq_rec_token_name']

data[features_null_col] = KNNImputer(n_neighbors=10).fit_transform(data[features_null_col])
print("填充后的缺失情况：", data.isnull().sum())


# 绘制特征ERC20_most_sent_token_type的词云图
from wordcloud import WordCloud

# 将特征转换为列表
ERC20=data['ERC20_most_sent_token_type']
ERC20=ERC20.apply(lambda x: str(x))
print(ERC20.dtype)
ERC20_Text=list(ERC20)
print(ERC20_Text)
#
# 云图
wc=WordCloud(max_words=1000, width=1600, height=1000, collocations=False).generate(" ".join(ERC20_Text))
plt.figure(figsize=(20, 20))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.imshow(wc)
plt.savefig('Fig.2(b).jpg', bbox_inches='tight',pad_inches=0,dpi=1500,)
plt.show()

# 保存数据
data.to_csv(path_or_buf=r'Data/Data_2.csv', index=None)