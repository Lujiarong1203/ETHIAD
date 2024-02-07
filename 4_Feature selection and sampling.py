import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
from sklearn.feature_selection import RFECV, RFE, SelectKBest
from sklearn.feature_selection import mutual_info_classif as MIC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from Unit_LJR import model_comparison_with_split
from imbalanced_ensemble.utils._plot import plot_2Dprojection_and_cardinality

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data= pd.read_csv("Data/Data_3.csv")
print(data.shape)

random_seed=1234

X=data.drop('FLAG', axis=1)
y=data['FLAG']
print(X.shape, y.shape)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_seed)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("训练集和测试集中标签比例：", Counter(y_train), Counter(y_test))   # 训练集{0: 6881, 1: 1975}，测试集{0: 781, 1: 204}

# 训练集中正常样本和欺诈样本比例约为7:3，因此对训练集进行样本采样
# 将不同采样方法写入k-v字典中
sampling_motheds={"SMOTE": SMOTE(random_state=random_seed), "ADASYS": ADASYN(random_state=random_seed),
                  "Tomek_Link": TomekLinks(), "SMOTE_Tomek_Link": SMOTETomek(random_state=random_seed),
                  }

# for meth_name, samp in sampling_motheds.items():
#     # 先进行样本采样
#     print(meth_name)
#     X_train_sampling, y_train_sampling=samp.fit_resample(X_train, y_train)
#     print('采样后训练集标签比例:', Counter(y_train_sampling))
#
#     # 再进行Lasso特征选择
#     lasso_model = LassoCV(alphas=[0.1, 1, 0.001, 0.0005], random_state=random_seed).fit(X_train_sampling, y_train_sampling)
#     print(lasso_model.alpha_)  # 模型所选择的最优正则化参数alpha
#
#     # 输出看模型最终选择了几个特征向量，剔除了几个特征向量
#     coef = pd.Series(lasso_model.coef_, index=X_train.columns)
#     print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")
#
#     # 索引和重要性做成dataframe形式
#     FI_lasso = pd.DataFrame({"Feature Importance": lasso_model.coef_}, index=X_train.columns)
#
#     # 由高到低进行排序
#     FI_lasso.sort_values("Feature Importance", ascending=False).round(3)
#
#     # 获取重要程度大于0的系数指标
#     FI_lasso[FI_lasso["Feature Importance"] != 0].sort_values("Feature Importance").plot(kind="barh",
#                                                                                          color='cornflowerblue',
#                                                                                          alpha=0.8)
#     plt.xticks(rotation=0)
#     plt.yticks(rotation=45)
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.xlabel('Feature coefficient', fontsize=11)
#     plt.ylabel('Feature name', fontsize=11)
#     plt.tick_params(labelsize=11)
#     # plt.savefig('Fig.6.jpg', dpi=700, bbox_inches='tight', pad_inches=0)  # 解决图片不清晰，不完整的问题
#     plt.show()
#
#     # Lasso特征选择后的 训练集 和 测试集
#     drop_colums = coef.index[abs(coef.values) == 0]
#     X_train_sampling_lasso = X_train_sampling.drop(drop_colums, axis=1)
#     X_test_lasso = X_test.drop(drop_colums, axis=1)
#     print('Lasso特征选择后的训练集和特征集的维度', X_train_sampling_lasso.shape, X_test_lasso.shape)
#
#     # 输出模型的性能
#     model_comparison_with_split(X_train_sampling_lasso, y_train_sampling, X_test_lasso, y_test,
#                                 XGBClassifier(random_state=random_seed), name=meth_name)

# for meth_name, samp in sampling_motheds.items():
#     # 先进行样本采样
#     print(meth_name)
#     X_train_sampling, y_train_sampling=samp.fit_resample(X_train, y_train)
#     print('采样后训练集标签比例:', Counter(y_train_sampling))
#
#     # 再进行MIC(互信息分类)特征选择
#     k_best = 50
#     mic_model = SelectKBest(MIC, k=k_best)
#     X_mic = mic_model.fit_transform(X_train_sampling, y_train_sampling)
#     mic_scores = mic_model.scores_
#     mic_indices = np.argsort(mic_scores)[::-1]
#     mic_k_best_features = list(X_train.columns.values[mic_indices[0:k_best]])
#     FI_mic = pd.DataFrame({"Feature Importance": mic_scores}, index=X_train.columns)
#     FI_mic[FI_mic["Feature Importance"] != 0].sort_values("Feature Importance").plot(kind="barh", color='firebrick',
#                                                                                      alpha=0.8)
#     plt.xticks(rotation=0, fontsize=11)
#     plt.xlabel('特征重要程度', fontsize=11)
#     plt.ylabel('特征名称', fontsize=11)
#     plt.show()
#
#     # MIC特征选择后的 训练集 和 测试集
#     X_train_sampling_MIC = X_train_sampling[mic_k_best_features]
#     X_test_MIC = X_test[mic_k_best_features]
#     print('MIC特征选择后的训练集和特征集的维度', X_train_sampling_MIC.shape, X_test_MIC.shape)
#
#     model_comparison_with_split(X_train_sampling_MIC, y_train_sampling, X_test_MIC, y_test,
#                                 XGBClassifier(random_state=random_seed), name=meth_name)

# for meth_name, samp in sampling_motheds.items():
#     # 先进行样本采样
#     print(meth_name)
#     X_train_sampling, y_train_sampling=samp.fit_resample(X_train, y_train)
#     print('采样后训练集标签比例:', Counter(y_train_sampling))
#
#     # RFECV递归特征消除法
#     rfe_model = RFECV(RandomForestClassifier(random_state=random_seed), cv=3, step=10, scoring='accuracy')
#     rfe = rfe_model.fit(X_train_sampling, y_train_sampling)
#     X_train_sampling_rfe = rfe.transform(X_train)  # 最优特征
#     # # RFE特征选择后的 训练集 和 测试集
#     X_train_sampling_RFE = X_train_sampling[rfe.get_feature_names_out()]
#     X_test_RFE = X_test[rfe.get_feature_names_out()]
#     print('RFE特征选择后的训练集和特征集的维度', X_train_sampling_RFE.shape, X_test_RFE.shape)
#
#     feature_ranking = rfe.ranking_
#     print(feature_ranking)
#     feature_importance_values = rfe.estimator_.feature_importances_
#     print(feature_importance_values)
#
#     FI_RFE = pd.DataFrame(feature_importance_values, index=X_train_sampling_RFE.columns, columns=['features importance'])
#     print(FI_RFE)
#
#     ## 由高到低进行排序
#     FI_RFE = FI_RFE.sort_values("features importance", ascending=False).round(3)
#     print(FI_RFE)
#
#     # 获取重要程度大于0的系数指标
#     plt.figure(figsize=(15, 10))
#     FI_RFE[FI_RFE["features importance"] != 0].sort_values("features importance").plot(kind="barh", color='firebrick',
#                                                                                        alpha=0.8)
#     plt.xticks(rotation=0)  # rotation代表lable显示的旋转角度，fontsize代表字体大小
#     plt.yticks(rotation=30)
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.xlabel('Feature importance', fontsize=15)
#     plt.ylabel('Feature name', fontsize=15)
#     plt.tick_params(labelsize=11)
#     # plt.savefig('Fig.5(d).jpg', dpi=700, bbox_inches='tight', pad_inches=0)  # 解决图片不清晰，不完整的问题
#     plt.show()
#
#     model_comparison_with_split(X_train_sampling_RFE, y_train_sampling, X_test_RFE, y_test,
#                                 XGBClassifier(random_state=random_seed), name=meth_name)


# 比较可得，经过ADASYS+Lasso特征选择后的模型的性能最佳，因此保存数据集
X_train_sampling, y_train_sampling=ADASYN(random_state=random_seed).fit_resample(X_train, y_train)
print('采样后训练集标签比例:', Counter(y_train_sampling))
# 再进行Lasso特征选择
lasso_model = LassoCV(alphas=[0.1, 1, 0.001, 0.0005], random_state=random_seed).fit(X_train_sampling, y_train_sampling)
print(lasso_model.alpha_)  # 模型所选择的最优正则化参数alpha
# 输出看模型最终选择了几个特征向量，剔除了几个特征向量
coef = pd.Series(lasso_model.coef_, index=X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")
# 索引和重要性做成dataframe形式
FI_lasso = pd.DataFrame({"Feature Importance": lasso_model.coef_}, index=X_train.columns)
# 由高到低进行排序
FI_lasso.sort_values("Feature Importance", ascending=False).round(3)
# 获取重要程度大于0的系数指标
FI_lasso[FI_lasso["Feature Importance"].abs()>0.005].sort_values("Feature Importance").plot(kind="barh", color='cornflowerblue', alpha=0.8)
plt.xticks(rotation=0)
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature coefficient', fontsize=11)
plt.ylabel('Feature name', fontsize=11)
plt.tick_params(labelsize=11)
# plt.savefig('Fig.6.jpg', dpi=700, bbox_inches='tight', pad_inches=0)  # 解决图片不清晰，不完整的问题
plt.show()
#
# Lasso特征选择后的 训练集 和 测试集
drop_colums = coef.index[abs(coef.values) == 0]
X_train_sampling_lasso = X_train_sampling.drop(drop_colums, axis=1)
X_test_lasso = X_test.drop(drop_colums, axis=1)
print('Lasso特征选择后的训练集和特征集的维度', X_train_sampling_lasso.shape, X_test_lasso.shape)
# 输出模型的性能
model_comparison_with_split(X_train_sampling_lasso, y_train_sampling, X_test_lasso, y_test,
                            XGBClassifier(random_state=random_seed), name="ADASYS+Lasso")
#
# 绘制数据 “采样+特征选择” 前后的降维2D分布
plot_2Dprojection_and_cardinality(X, y)
plt.tick_params(labelsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xticks([0, 1], ['Normal', 'Illicit'], rotation='horizontal')
plt.legend(loc='upper right')
plt.show()

# 分别合并训练集和测试集
ori_train_set=pd.concat([X_train, y_train], axis=1)
train_set=pd.concat([X_train_sampling_lasso, y_train_sampling], axis=1)
test_set=pd.concat([X_test_lasso, y_test], axis=1)

# 绘制TSNE降维图
from Unit_LJR import TSNE_plot
TSNE_plot(data)


# 分别保存训练集和测试集
train_set=pd.concat([X_train_sampling_lasso, y_train_sampling], axis=1)
test_set=pd.concat([X_test_lasso, y_test], axis=1)

train_set.to_csv(path_or_buf=r'Data/train_set.csv', index=None)
test_set.to_csv(path_or_buf=r'Data/test_set.csv', index=None)