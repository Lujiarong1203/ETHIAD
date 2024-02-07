import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import scikitplot as skplt

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

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
# 准备比较的模型
# LR
LR=LogisticRegression(random_state=random_seed)
LR.fit(X_train, y_train)
y_pred_LR=LR.predict(X_test)
y_proba_LR=LR.predict_proba(X_test)
cm_LR=confusion_matrix(y_test, y_pred_LR)
#
# SVM
SVM=SVC(probability=True, random_state=random_seed)
SVM.fit(X_train, y_train)
y_pred_SVM=SVM.predict(X_test)
y_proba_SVM=SVM.predict_proba(X_test)
cm_SVM=confusion_matrix(y_test, y_pred_SVM)

# MLP
MLP=MLPClassifier(random_state=random_seed)
MLP.fit(X_train, y_train)
y_pred_MLP=MLP.predict(X_test)
y_proba_MLP=MLP.predict_proba(X_test)
cm_MLP=confusion_matrix(y_test, y_pred_MLP)

# SGD
SGD=SGDClassifier(loss="log", random_state=random_seed)
SGD.fit(X_train, y_train)
y_pred_SGD=SGD.predict(X_test)
y_proba_SGD=SGD.predict_proba(X_test)
cm_SGD=confusion_matrix(y_test, y_pred_SGD)

# BNB
BNB=BernoulliNB()
BNB.fit(X_train, y_train)
y_pred_BNB=BNB.predict(X_test)
y_proba_BNB=BNB.predict_proba(X_test)
cm_BNB=confusion_matrix(y_test, y_pred_BNB)

# GNB
GNB=GaussianNB()
GNB.fit(X_train, y_train)
y_pred_GNB=GNB.predict(X_test)
y_proba_GNB=GNB.predict_proba(X_test)
cm_GNB=confusion_matrix(y_test, y_pred_GNB)

# DT
DT=DecisionTreeClassifier(random_state=random_seed)
DT.fit(X_train, y_train)
y_pred_DT=DT.predict(X_test)
y_proba_DT=DT.predict_proba(X_test)
cm_DT=confusion_matrix(y_test, y_pred_DT)

# RF
RF=RandomForestClassifier(random_state=random_seed)
RF.fit(X_train, y_train)
y_pred_RF=RF.predict(X_test)
y_proba_RF=RF.predict_proba(X_test)
cm_RF=confusion_matrix(y_test, y_pred_RF)

# Ada
Ada=AdaBoostClassifier(random_state=random_seed)
Ada.fit(X_train, y_train)
y_pred_Ada=Ada.predict(X_test)
y_proba_Ada=Ada.predict_proba(X_test)
cm_Ada=confusion_matrix(y_test, y_pred_Ada)

# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)
GBDT.fit(X_train, y_train)
y_pred_GBDT=GBDT.predict(X_test)
y_proba_GBDT=GBDT.predict_proba(X_test)
cm_GBDT=confusion_matrix(y_test, y_pred_GBDT)

# LGBM
LGBM=LGBMClassifier(random_state=random_seed)
LGBM.fit(X_train, y_train)
y_pred_LGBM=LGBM.predict(X_test)
y_proba_LGBM=LGBM.predict_proba(X_test)
cm_LGBM=confusion_matrix(y_test, y_pred_LGBM)

# KNN
KNN=KNeighborsClassifier()
KNN.fit(X_train, y_train)
y_pred_KNN=KNN.predict(X_test)
y_proba_KNN=KNN.predict_proba(X_test)
cm_KNN=confusion_matrix(y_test, y_pred_KNN)

# Catboost
Catboost=CatBoostClassifier(random_state=random_seed)
Catboost.fit(X_train, y_train)
y_pred_Catboost=Catboost.predict(X_test)
y_proba_Catboost=Catboost.predict_proba(X_test)
cm_Catboost=confusion_matrix(y_test, y_pred_Catboost)

# XGboost
XG=XGBClassifier(learning_rate=0.6, n_estimators=140, max_depth=6, subsample=1, colsample_bytree=1, min_child_weight=1, random_state=random_seed)
XG.fit(X_train, y_train)
y_pred_XG=XG.predict(X_test)
y_proba_XG=XG.predict_proba(X_test)
cm_XG=confusion_matrix(y_test, y_pred_XG)
acc_XG=accuracy_score(y_test, y_pred_XG)
pre_XG=precision_score(y_test, y_pred_XG)
rec_XG=recall_score(y_test, y_pred_XG)
f1_XG=f1_score(y_test, y_pred_XG)
AUC_XG=roc_auc_score(y_test, y_pred_XG)
print("XGboost的性能:", acc_XG, pre_XG, rec_XG, f1_XG, AUC_XG)

# 绘制多个模型的混淆矩阵图
# # 1-LR
# skplt.metrics.plot_confusion_matrix(y_test, y_pred_LR, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(1) LR', y=-0.2, fontsize=15)
# plt.xlabel('Predicted value', fontsize=15)
# plt.ylabel('True Value', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.show()
#
# # 2-KNN
# skplt.metrics.plot_confusion_matrix(y_test, y_pred_KNN, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(2) KNN', y=-0.2, fontsize=15)
# plt.xlabel('Predicted value', fontsize=15)
# plt.ylabel('True Value', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.show()
#
# # 3-SVM
# skplt.metrics.plot_confusion_matrix(y_test, y_pred_SVM, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(3) SVM', y=-0.2, fontsize=15)
# plt.xlabel('Predicted value', fontsize=15)
# plt.ylabel('True Value', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.show()
#
# # 4-DT
# skplt.metrics.plot_confusion_matrix(y_test, y_pred_DT, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(4) DT', y=-0.2, fontsize=15)
# plt.xlabel('Predicted value', fontsize=15)
# plt.ylabel('True Value', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.show()
#
# # 5-RF
# skplt.metrics.plot_confusion_matrix(y_test, y_pred_RF, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(5) RF', y=-0.2, fontsize=15)
# plt.xlabel('Predicted value', fontsize=15)
# plt.ylabel('True Value', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.show()

# 6-GBDT
skplt.metrics.plot_confusion_matrix(y_test, y_pred_GBDT, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(a) GBDT', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

# # 7-LGBM
# skplt.metrics.plot_confusion_matrix(y_test, y_pred_LGBM, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(7) LGBM', y=-0.2, fontsize=15)
# plt.xlabel('Predicted value', fontsize=15)
# plt.ylabel('True Value', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.show()

# 8-Adaboost
skplt.metrics.plot_confusion_matrix(y_test, y_pred_Ada, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(b) Adaboost', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

# # 9-Catboost
# skplt.metrics.plot_confusion_matrix(y_test, y_pred_Catboost, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(9) Catboost', y=-0.2, fontsize=15)
# plt.xlabel('Predicted value', fontsize=15)
# plt.ylabel('True Value', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.show()

# 10-XGboost
skplt.metrics.plot_confusion_matrix(y_test, y_pred_XG, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(c) ETHIAD', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

# XGboost KS Curve
skplt.metrics.plot_ks_statistic(y_test, y_proba_XG, title=None, text_fontsize=15, figsize=(6, 6))
# plt.title('(a) IICOFDM', y=-0.2, fontsize=15)
plt.legend(fontsize=15, loc='lower right')
plt.savefig('Fig.6(b).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# XGboost cumulative_gain Curve
skplt.metrics.plot_cumulative_gain(y_test, y_proba_XG, title=None, text_fontsize=15, figsize=(6, 6))
plt.legend(loc='lower right', fontsize=15)
plt.savefig('Fig.6(c).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# XGboost Lift Curve
skplt.metrics.plot_lift_curve(y_test, y_proba_XG, title=None, text_fontsize=15, figsize=(6, 6))
plt.legend(loc='upper right', fontsize=15)
plt.savefig('Fig.6(d).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# 多个模型的ROC曲线对比
plt.rc('font',family='Times New Roman')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif']=['SimHei']
fpr1, tpr1, thres1 = roc_curve(y_test, y_proba_LR[:, 1])
fpr2, tpr2, thres2 = roc_curve(y_test, y_proba_SVM[:, 1])
fpr3, tpr3, thres3 = roc_curve(y_test, y_proba_KNN[:,1])
fpr4, tpr4, thres4 = roc_curve(y_test, y_proba_DT[:, 1])
fpr5, tpr5, thres5 = roc_curve(y_test, y_proba_RF[:, 1])
fpr6, tpr6, thres6 = roc_curve(y_test, y_proba_Catboost[:, 1])
fpr7, tpr7, thres7 = roc_curve(y_test, y_proba_GBDT[:, 1])
fpr8, tpr8, thres8 = roc_curve(y_test, y_proba_Ada[:, 1])
fpr9, tpr9, thres9 = roc_curve(y_test, y_proba_LGBM[:, 1])
fpr10, tpr10, thres10 = roc_curve(y_test, y_proba_XG[:, 1])


plt.figure(figsize=(6, 6))
plt.grid()
plt.plot(fpr1, tpr1, 'b', label='LR ', color='k',lw=1.5,ls='--')
plt.plot(fpr2, tpr2, 'b', label='SVM ', color='darkorange',lw=1.5,ls='--')
plt.plot(fpr3, tpr3, 'b', label='KNN ', color='peru',lw=1.5,ls='--')
plt.plot(fpr4, tpr4, 'b', label='DT ', color='lime',lw=1.5,ls='--')
plt.plot(fpr5, tpr5, 'b', label='RF ', color='fuchsia',lw=1.5,ls='--')

plt.plot(fpr6, tpr6, 'b', label='ETC ', color='cyan',lw=1.5,ls='--')
plt.plot(fpr7, tpr7, 'b', label='GBDT ', color='green',lw=1.5,ls='--')
plt.plot(fpr8, tpr8, 'b', label='Adaboost ', color='blue',lw=1.5,ls='--')
plt.plot(fpr9, tpr9, 'b', label='LightGBM ', color='violet',lw=1.5, ls='--')
plt.plot(fpr10, tpr10, 'b', ms=1,label='XGboost ', lw=3.5,color='red',marker='*')

plt.plot([0, 1], [0, 1], 'darkgrey')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.ylabel('True Positive Rate', fontname='Times New Roman', fontsize=15)
plt.xlabel('False Positive Rate', fontname='Times New Roman', fontsize=15)
plt.tick_params(labelsize=12)
plt.legend(fontsize=15)
plt.savefig('Fig.6(a).jpg', dpi=700, bbox_inches='tight',pad_inches=0)
plt.show()


