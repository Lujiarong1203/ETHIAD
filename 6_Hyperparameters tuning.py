import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
# 超参数调优
# 1-learning_rate
cv_params= {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
model = XGBClassifier(random_state=random_seed)
optimized_XG = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=10, verbose=1, n_jobs=-1)
optimized_XG.fit(X_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_XG.best_params_))
print('Best model score:{0}'.format(optimized_XG.best_score_))

# Draw the learning_rate validation_curve
param_range_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=X_train,
                                             y=y_train,
                                             param_name='learning_rate',
                                             param_range=param_range_1,
                                             cv=10, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

print(train_scores_1, '\n', train_mean_1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label="Training score")

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10,label="Cross-validation score")

plt.fill_between(param_range_1,test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(a) learning_rate', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.9925, 1.0])
plt.tight_layout()
plt.show()

# 2-n_estimators
cv_params= {'n_estimators': range(90, 160, 10)}
model = XGBClassifier(learning_rate=0.6, random_state=random_seed)
optimized_XG = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=10, verbose=1, n_jobs=-1)
optimized_XG.fit(X_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_XG.best_params_))
print('Best model score:{0}'.format(optimized_XG.best_score_))

# Draw the n_estimators validation_curve
param_range_1=range(90, 160, 10)
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=X_train,
                                             y=y_train,
                                             param_name='n_estimators',
                                             param_range=param_range_1,
                                             cv=10, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

print(train_scores_1, '\n', train_mean_1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='Training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10,label='Cross-validation score')

plt.fill_between(param_range_1,test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(b) n_estimators', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.9925, 1.0])
plt.tight_layout()
plt.show()


# 3-max_depth
cv_params= {'max_depth': range(1, 10, 1)}
model = XGBClassifier(learning_rate=0.6, n_estimators=140, random_state=random_seed)
optimized_XG = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=10, verbose=1, n_jobs=-1)
optimized_XG.fit(X_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_XG.best_params_))
print('Best model score:{0}'.format(optimized_XG.best_score_))

# Draw the max_depth validation curve
param_range_1=range(1, 10, 1)
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=X_train,
                                             y=y_train,
                                             param_name='max_depth',
                                             param_range=param_range_1,
                                             cv=10, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='Training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='Cross-validation score')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(c) max_depth', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.9925, 1.0])
plt.tight_layout()
plt.show()


# 4-subsample
cv_params= {'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
model = XGBClassifier(learning_rate=0.6, n_estimators=140, max_depth=6, random_state=random_seed)
optimized_XG = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=10, verbose=1, n_jobs=-1)
optimized_XG.fit(X_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_XG.best_params_))
print('Best model score:{0}'.format(optimized_XG.best_score_))

# Draw the max_depth validation curve
param_range_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=X_train,
                                             y=y_train,
                                             param_name='subsample',
                                             param_range=param_range_1,
                                             cv=10, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='Training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='Cross-validation score')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(d) subsample', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.99, 1.0])
plt.tight_layout()
plt.show()


# 5-colsample_bytree
cv_params= {'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
model = XGBClassifier(learning_rate=0.6, n_estimators=140, max_depth=6, subsample=1, random_state=random_seed)
optimized_XG = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=10, verbose=1, n_jobs=-1)
optimized_XG.fit(X_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_XG.best_params_))
print('Best model score:{0}'.format(optimized_XG.best_score_))

# Draw the max_depth validation curve
param_range_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=X_train,
                                             y=y_train,
                                             param_name='colsample_bytree',
                                             param_range=param_range_1,
                                             cv=10, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='Training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='Cross-validation score')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(e) colsample_bytree', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.99, 1.0])
plt.tight_layout()
plt.show()

# 6-min_child_weight
cv_params= {'min_child_weight': [0.008, 0.009, 0.01, 0.02, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9 ,1]}
model = XGBClassifier(learning_rate=0.6, n_estimators=140, max_depth=6, subsample=1, colsample_bytree=1, random_state=random_seed)
optimized_XG = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=10, verbose=1, n_jobs=-1)
optimized_XG.fit(X_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_XG.best_params_))
print('Best model score:{0}'.format(optimized_XG.best_score_))

# Draw the max_depth validation curve
param_range_1 = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=X_train,
                                             y=y_train,
                                             param_name='min_child_weight',
                                             param_range=param_range_1,
                                             cv=10, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='Training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='Cross-validation score')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(f) min_child_weight', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.99, 1.0])
plt.tight_layout()
plt.show()


from Unit_LJR import model_comparison_with_split
model_comparison_with_split(X_train, y_train, X_test, y_test, estimator=XGBClassifier(random_state=random_seed), name="Before_Tuning")
model_comparison_with_split(X_train, y_train, X_test, y_test,
                            estimator=XGBClassifier(
                                learning_rate=0.6,
                                n_estimators=140,
                                max_depth=6,
                                subsample=1,
                                colsample_bytree=1,
                                min_child_weight=1,
                                random_state=random_seed
                            ),
                            name="After_Tuning")
