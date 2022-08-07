#%% - nạp thư viện
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import statsmodels.api as sm
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
#%% - Load dữ liệu
df = pd.read_csv('./Data/DL.csv')
df.head()
df.info
df.describe(include='all')
#%% - Kiểm tra missing value
print(df.isna().values.any())
# Verification
print(df.isna().sum())
#Kiểm tra giá trị trùng lặp
print(df.duplicated().value_counts())
#Xóa khoảng trắng
df.columns = df.columns.str.replace(' ', '')
#Khám phá dữ liệu
df.groupby('y').mean()
#%% - Hệ số tương quan Pearson
print(df.corr())
#%% - kiểm định sự tương quan giữa 2 biến phân loại bằng kiểm định chi-square:
# CrosstabResult = pd.crosstab(index=df['housing'], columns=df['y'])
# print(CrosstabResult)
# from scipy.stats import chi2_contingency
# ChiSqResult = chi2_contingency(CrosstabResult)
# print('The P-Value of the ChiSq Test is:', ChiSqResult[1])
#==============================
#%%
#%% - Tạo cột mới, xóa cột cũ
new_df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'month', 'loan', 'contact', 'poutcome'], drop_first=True)
new_df.head()
# Chuyển đổi dữ liệu
new_df['y'].replace('yes', 1, inplace=True)
new_df['y'].replace('no', 0, inplace=True)
new_df.head()
#
print(new_df.corr())
#%% - Chia dữ liệu train/test ( 75% train và 25% test)
#%%
x = new_df.drop("y", axis='columns')
y = new_df["y"]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0, train_size=.75)
#%% - Xây dựng model
logit_model = sm.Logit(Y_train, X_train)
result = logit_model.fit(method='newton')
print(result.summary())

lgt_model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
lgt_model.fit(X_train, Y_train)
y_pred = lgt_model.predict(X_test)
print(lgt_model.coef_, lgt_model.intercept_)

print(confusion_matrix(Y_test, y_pred))
print(confusion_matrix(y_pred, Y_test))
print(classification_report(Y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
print("Precision:", metrics.precision_score(Y_test, y_pred))
print("Recall:", metrics.recall_score(Y_test, y_pred))
print('Cross Validation mean:', (cross_val_score(lgt_model, X_train, Y_train, cv=5, n_jobs=2, scoring='accuracy').mean()))

#%% - Confusion Matrix
cm = confusion_matrix(Y_test, y_pred)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative', 'Positive']
plt.title('Ma trận nhầm lẫn')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()
#=========
cm = confusion_matrix(Y_test, y_pred)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
#=============
y_train_pred = lgt_model.predict(X_train)
print(classification_report(Y_train, y_train_pred))
cm = confusion_matrix(Y_train, y_train_pred)
print(cm)
#%%
fig, ax = plt.subplots()
ax.imshow(cm)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='r', fontsize=22)
plt.show()
#%% - ROC
ROS = RandomOverSampler(sampling_strategy='minority', random_state=1)
X_train_ROS, y_train_ROS = ROS.fit_resample(X_train, Y_train)
np.bincount(y_train_ROS)
lgt_oversampling = LogisticRegression(solver='liblinear')
lgt_oversampling.fit(X_train_ROS, y_train_ROS)
y_pred_oversampling = lgt_oversampling.predict(X_test)

y_pred_proba = lgt_oversampling.predict_proba(X_test)[:, 1]
sns.set(rc={'figure.figsize': (6, 5)})
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#%% - Mô hình đơn biến
# x = new_df.drop("y", axis='columns')
# y = new_df["y"]
# uni_x = df['housing']
# uni_y = df['y']
# df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
uni_x = new_df['housing_yes']
uni_y = new_df['y']
x_train, x_test, y_train, y_test = train_test_split(uni_x, uni_y, test_size=.75)

uni_model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(x_train, y_train)
#In thông số
intercept = uni_model.intercept_
coefs = uni_model.coef_
score = uni_model.score(x_train, y_train)
prob_matrix = uni_model.predict_proba(x_train)
#%% - Ma trận nhầm lẫn
y_uni_pred = uni_model.predict(x_train)
print(classification_report(y_train, y_uni_pred))
cm_uni = confusion_matrix(y_train, y_uni_pred)
print(cm_uni)
fig, ax = plt.subplots()
ax.imshow(cm_uni)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm_uni[i, j], ha='center', va='center', color='r', fontsize=22)
plt.show()