import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import shap

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

# 标准化采样后的训练集
mm = StandardScaler()
X_train_std = pd.DataFrame(mm.fit_transform(X_train))
X_train_std.columns = X_train.columns

y_train=train_data['FLAG']

X_test=test_data.drop('FLAG', axis=1)

# 标准化采样后的训练集
mm = StandardScaler()
X_test_std = pd.DataFrame(mm.fit_transform(X_test))
X_test_std.columns = X_test.columns

y_test=test_data['FLAG']
print('划分特征集和标签集后训练集和测试集的维度:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print('训练集和测试集中标签比例：', Counter(y_train), Counter(y_test))

random_seed=1234

# RF
RF=RandomForestClassifier(random_state=random_seed)
RF.fit(X_train, y_train)
y_pred_RF=RF.predict(X_test)
y_proba_RF=RF.predict_proba(X_test)
cm_RF=confusion_matrix(y_test, y_pred_RF)
print('RF:', '\n', cm_RF)

# LGBM
LGBM=LGBMClassifier(random_state=random_seed)
LGBM.fit(X_train, y_train)
y_pred_LGBM=LGBM.predict(X_test)
y_proba_LGBM=LGBM.predict_proba(X_test)
cm_LGBM=confusion_matrix(y_test, y_pred_LGBM)
print('LGBM:', '\n', cm_LGBM)

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
print("XGboost的性能:", acc_XG, pre_XG, rec_XG, f1_XG, AUC_XG, '\n', cm_XG)


# # 输出特征重要性图
# RF 重要性
RF_feature_importance = RF.feature_importances_
FI_RF=pd.DataFrame(RF_feature_importance, index=X_train.columns, columns=['features importance'])
FI_RF=FI_RF.sort_values("features importance",ascending=False)
print('FI_RF', FI_RF)

# LightGBM 重要性
LGBM_feature_importance = LGBM.feature_importances_
FI_LGBM=pd.DataFrame(LGBM_feature_importance, index=X_train.columns, columns=['features importance'])
FI_LGBM=FI_LGBM.sort_values("features importance",ascending=False)
print('FI_LGBM', FI_LGBM)

# XGboost 重要性
XG_feature_importance =XG.feature_importances_
FI_XG=pd.DataFrame(XG_feature_importance, index=X_train.columns, columns=['features importance'])
FI_XG=FI_XG.sort_values("features importance",ascending=False)
print('FI_XG', FI_XG)

# SHAP  重要性
explainer = shap.TreeExplainer(XG)
shap_value = explainer.shap_values(X_train)
print('SHAP值：', shap_value)
print('期望值：', explainer.expected_value)

SHAP_feature_importance = np.abs(shap_value).mean(0)
print(SHAP_feature_importance)

FI_SHAP=pd.DataFrame(SHAP_feature_importance, index=X_train.columns, columns=['features importance'])
FI_SHAP=FI_SHAP.sort_values("features importance",ascending=False)
print('FI_SHAP', FI_SHAP)


# """
# 绘制特征重要性图
# """

# 绘制RF的重要性图-1
FI_RF.head(20).sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0)
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=10)
plt.ylabel('Feature name',fontsize=10)
plt.tick_params(labelsize = 10)
# plt.title('RF')
plt.savefig('Fig.7(a).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()

# 绘制LGBM的重要性图-2
FI_LGBM.head(20).sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0)
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=10)
plt.ylabel('Feature name',fontsize=10)
plt.tick_params(labelsize = 10)
# plt.title('LGBM')
plt.savefig('Fig.7(b).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()

# 绘制XGboost的重要性图-3
FI_XG.head(20).sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0)
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=10)
plt.ylabel('Feature name',fontsize=10)
plt.tick_params(labelsize = 10)
# plt.title('XGboost')
plt.savefig('Fig.7(c).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()

# 绘制SHAP的重要性图-4
FI_SHAP.head(20).sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0)
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=10)
plt.ylabel('Feature name',fontsize=10)
plt.tick_params(labelsize = 10)
# plt.title('SHAP')
plt.savefig('Fig.7(d).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()

# SHAP summary plot
# fig = plt.subplots(figsize=(6,4),dpi=400)   plot_type="dot",
ax=shap.summary_plot(shap_value, X_train, max_display=20)

# SHAP dependence plot
# 第一组
shap.dependence_plot("Time_Diff_between_first_and_last_(Mins)", shap_value, X_train, interaction_index="min_val_sent")

# 第二组
shap.dependence_plot("Unique_Received_From_Addresses", shap_value, X_train, interaction_index="min_val_sent")

# # SHAP force/waterfall/decision plot
# # non-fraudent

# for i in [152, 155, 162, 203, 221, 279]:
#     print("样本序号：", i)
#     shap.initjs()
#     shap.force_plot(explainer.expected_value,
#                     shap_value[i],
#                     X_train.iloc[i],
#                     text_rotation=20,
#                     matplotlib=True)


shap.initjs()
shap.force_plot(explainer.expected_value,
                shap_value[221],
                X_train.iloc[221],
                text_rotation=20,
                matplotlib=True)

shap.plots._waterfall.waterfall_legacy(explainer.expected_value,
                                       shap_value[221],
                                       feature_names = X_train.columns,
                                       max_display = 19)

shap.decision_plot(explainer.expected_value,
                   shap_value[221],
                   X_train.iloc[221])

