import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import scikitplot as skplt

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

random_seed=1234
# 准备模型
XG=XGBClassifier(learning_rate=0.6, n_estimators=140, max_depth=6, subsample=1, colsample_bytree=1, min_child_weight=1, random_state=random_seed)

Catboost=CatBoostClassifier(random_state=random_seed)

RF=RandomForestClassifier(random_state=random_seed)

LGBM=LGBMClassifier(random_state=random_seed)

LR=LogisticRegression(random_state=random_seed)

GBDT=GradientBoostingClassifier(random_state=random_seed)

KNN=KNeighborsClassifier()

SVM=SVC(random_state=random_seed)

# 绘制学习曲线
# 1_LR
# skplt.estimators.plot_learning_curve(LR, X_train, y_train, title=None, cv=10, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
# plt.legend(loc='lower right', fontsize=15)
# plt.xlabel('Training sample size', fontsize=15)
# plt.ylabel('Score', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.title('(a) LR', y=-0.2, fontproperties='Times New Roman', fontsize=15)
# plt.tight_layout()
# plt.show()
#
# # 2_SVM
# skplt.estimators.plot_learning_curve(SVM, X_train, y_train, title=None, cv=10, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
# plt.legend(loc='lower right', fontsize=15)
# plt.xlabel('Training sample size', fontsize=15)
# plt.ylabel('Score', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.title('(b) SVM', y=-0.2, fontproperties='Times New Roman', fontsize=15)
# plt.tight_layout()
# plt.show()
#
# # 3_KNN
# skplt.estimators.plot_learning_curve(KNN, X_train, y_train, title=None, cv=10, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
# plt.legend(loc='lower right', fontsize=15)
# plt.xlabel('Training sample size', fontsize=15)
# plt.ylabel('Score', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.title('(c) KNN', y=-0.2, fontproperties='Times New Roman', fontsize=15)
# plt.tight_layout()
# plt.show()

# 4_RF
# skplt.estimators.plot_learning_curve(RF, X_train, y_train, title=None, cv=5, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
# plt.legend(loc='lower right', fontsize=15)
# plt.xlabel('Training sample size', fontsize=15)
# plt.ylabel('Score', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.title('(d) RF', y=-0.2, fontproperties='Times New Roman', fontsize=15)
# plt.tight_layout()
# plt.show()

# 5_LGBM
# skplt.estimators.plot_learning_curve(LGBM, X_train, y_train, title=None, cv=5, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
# plt.legend(loc='lower right', fontsize=15)
# plt.xlabel('Training sample size', fontsize=15)
# plt.ylabel('Score', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.title('(e) LGBM', y=-0.2, fontproperties='Times New Roman', fontsize=15)
# plt.tight_layout()
# plt.show()

# 6_GBDT
skplt.estimators.plot_learning_curve(GBDT, X_train, y_train, title=None, cv=5, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training sample size', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(a) GBDT', y=-0.2, fontproperties='Times New Roman', fontsize=15)
plt.tight_layout()
plt.show()

# # 7_Catboost
# skplt.estimators.plot_learning_curve(Catboost, X_train, y_train, title=None, cv=10, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
# plt.legend(loc='lower right', fontsize=15)
# plt.xlabel('Training sample size', fontsize=15)
# plt.ylabel('Score', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.title('(g) Catboost', y=-0.2, fontproperties='Times New Roman', fontsize=15)
# plt.tight_layout()
# plt.show()

# 8_ETHIAD
skplt.estimators.plot_learning_curve(XG, X_train, y_train, title=None, cv=5, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training sample size', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(b) ETHIAD', y=-0.2, fontproperties='Times New Roman', fontsize=15)
plt.tight_layout()
plt.show()


