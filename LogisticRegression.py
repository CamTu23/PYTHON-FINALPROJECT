#%% - nạp thư viện
import numpy as np
import pandas as pd
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
#%% - Trực quan hóa dữ liệu
# sns.pairplot(df)
sns.pairplot(df, hue='y')
plt.show()
#
#%% - Thông số để điều chỉnh graph
plt.rcParams['figure.figsize'] = (10, 9)
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 13
#Biến phân loại: ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]
#Education
sns.countplot(x="education", data=df, hue="y", order=df["education"].value_counts().index)
plt.title("Mối quan hệ giữa education và y ")
plt.show()
#Marital
sns.countplot(x="marital", data=df, hue="y", order=df["marital"].value_counts().index)
plt.title("Mối quan hệ giữa Martial và y")
plt.show()
#Job
sns.countplot(x="job", data=df, hue="y", order=df["job"].value_counts().index)
plt.title("Mối quan hệ giữa Job và y")
plt.xticks(rotation=30)
plt.show()
#Loan
sns.countplot(x="loan", data=df, hue="y", order=df["loan"].value_counts().index)
plt.title("Mối quan hệ giữa loan và y")
plt.show()
#Housing
sns.countplot(x="housing", data=df, hue="y", order=df["housing"].value_counts().index)
plt.title(" Mối quan hệ gữa housing và y")
plt.show()
#Contact
sns.countplot(x="contact", data=df, hue="y", order=df["contact"].value_counts().index)
plt.title(" Mối quan hệ giữa Contact và y")
plt.show()
# Numerical Features: ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
#%% - set thông số chung cho biểu đồ
# Thiết lập màu nền (sử dụng sns.set_theme() nếu seaborn version 0.11.0 or above)
sns.set(rc={'figure.figsize': (11, 8)}, font_scale=1.5, style='whitegrid')
# Tạo figure kết hợp của 2 biểu đồ box và hist với matplotlib.Axes objects (ax_box and ax_hist)
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.3, 1)})
#Age
# Thiết lập màu nền (sử dụng sns.set_theme() nếu seaborn version 0.11.0 or above)
sns.set(rc={'figure.figsize': (11, 8)}, font_scale=1.5, style='whitegrid')
# Tạo figure kết hợp của 2 biểu đồ box và hist với matplotlib.Axes objects (ax_box and ax_hist)
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.3, 1)})
#boxplot
mean = df['age'].mean()
median = df['age'].median()
mode = df['age'].mode().values[0]
# df.boxplot(column='age', by='y', figsize=(5, 6))
sns.boxplot(data=df, x="age", y="y", ax=ax_box, order=df["y"].value_counts().index)
ax_box.axvline(mean, color='r', linestyle='--')
ax_box.axvline(median, color='g', linestyle='-')
ax_box.axvline(mode, color='b', linestyle='-')
#histogram
sns.histplot(data=df, x="age", ax=ax_hist, kde=True)
ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
ax_hist.axvline(median, color='g', linestyle='-', label="Median")
ax_hist.axvline(mode, color='b', linestyle='-', label="Mode")
ax_hist.legend()
ax_box.set(xlabel='')
plt.show()
#balance
# Thiết lập màu nền (sử dụng sns.set_theme() nếu seaborn version 0.11.0 or above)
sns.set(rc={'figure.figsize': (11, 8)}, font_scale=1.5, style='whitegrid')
# Tạo figure kết hợp của 2 biểu đồ box và hist với matplotlib.Axes objects (ax_box and ax_hist)
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.3, 1)})

mean = df['balance'].mean()
median = df['balance'].median()
mode = df['balance'].mode().values[0]
sns.boxplot(data=df, x="balance", y="y", ax=ax_box, order=df["y"].value_counts().index)
ax_box.axvline(mean, color='r', linestyle='--')
ax_box.axvline(median, color='g', linestyle='-')
ax_box.axvline(mode, color='b', linestyle='-')
sns.histplot(data=df, x="balance", ax=ax_hist, kde=True)
ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
ax_hist.axvline(median, color='g', linestyle='-', label="Median")
ax_hist.axvline(mode, color='b', linestyle='-', label="Mode")
ax_hist.legend()
ax_box.set(xlabel='')
plt.show()
#day
# Thiết lập màu nền (sử dụng sns.set_theme() nếu seaborn version 0.11.0 or above)
sns.set(rc={'figure.figsize': (11, 8)}, font_scale=1.5, style='whitegrid')
# Tạo figure kết hợp của 2 biểu đồ box và hist với matplotlib.Axes objects (ax_box and ax_hist)
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.3, 1)})
mean = df['day'].mean()
median = df['day'].median()
mode = df['day'].mode().values[0]
sns.boxplot(data=df, x="day", y="y", ax=ax_box, order=df["y"].value_counts().index)
ax_box.axvline(mean, color='r', linestyle='--')
ax_box.axvline(median, color='g', linestyle='-')
ax_box.axvline(mode, color='b', linestyle='-')
sns.histplot(data=df, x="day", ax=ax_hist, kde=True)
ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
ax_hist.axvline(median, color='g', linestyle='-', label="Median")
ax_hist.axvline(mode, color='b', linestyle='-', label="Mode")
ax_hist.legend()
ax_box.set(xlabel='')
plt.show()
#duration
# Thiết lập màu nền (sử dụng sns.set_theme() nếu seaborn version 0.11.0 or above)
sns.set(rc={'figure.figsize': (11, 8)}, font_scale=1.5, style='whitegrid')
# Tạo figure kết hợp của 2 biểu đồ box và hist với matplotlib.Axes objects (ax_box and ax_hist)
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.3, 1)})

mean = df['duration'].mean()
median = df['duration'].median()
mode = df['duration'].mode().values[0]
sns.boxplot(data=df, x="duration", y="y", ax=ax_box, order=df["y"].value_counts().index)
ax_box.axvline(mean, color='r', linestyle='--')
ax_box.axvline(median, color='g', linestyle='-')
ax_box.axvline(mode, color='b', linestyle='-')

sns.histplot(data=df, x="duration", ax=ax_hist, kde=True)
ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
ax_hist.axvline(median, color='g', linestyle='-', label="Median")
ax_hist.axvline(mode, color='b', linestyle='-', label="Mode")
ax_hist.legend()
ax_box.set(xlabel='')
plt.show()
#campaign
# Thiết lập màu nền (sử dụng sns.set_theme() nếu seaborn version 0.11.0 or above)
sns.set(rc={'figure.figsize': (11, 8)}, font_scale=1.5, style='whitegrid')
# Tạo figure kết hợp của 2 biểu đồ box và hist với matplotlib.Axes objects (ax_box and ax_hist)
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.3, 1)})

mean = df['campaign'].mean()
median = df['campaign'].median()
mode = df['campaign'].mode().values[0]
sns.boxplot(data=df, x="campaign", y="y", ax=ax_box, order=df["y"].value_counts().index)
ax_box.axvline(mean, color='r', linestyle='--')
ax_box.axvline(median, color='g', linestyle='-')
ax_box.axvline(mode, color='b', linestyle='-')

sns.histplot(data=df, x="campaign", ax=ax_hist, kde=True)
ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
ax_hist.axvline(median, color='g', linestyle='-', label="Median")
ax_hist.axvline(mode, color='b', linestyle='-', label="Mode")
ax_hist.legend()
ax_box.set(xlabel='')
plt.show()
#pdays
# Thiết lập màu nền (sử dụng sns.set_theme() nếu seaborn version 0.11.0 or above)
sns.set(rc={'figure.figsize': (11, 8)}, font_scale=1.5, style='whitegrid')
# Tạo figure kết hợp của 2 biểu đồ box và hist với matplotlib.Axes objects (ax_box and ax_hist)
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.3, 1)})
mean = df['pdays'].mean()
median = df['pdays'].median()
mode = df['pdays'].mode().values[0]

sns.boxplot(data=df, x="pdays", y="y", ax=ax_box, order=df["y"].value_counts().index)
ax_box.axvline(mean, color='r', linestyle='--')
ax_box.axvline(median, color='g', linestyle='-')
ax_box.axvline(mode, color='b', linestyle='-')

sns.histplot(data=df, x="pdays", ax=ax_hist, kde=True)
ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
ax_hist.axvline(median, color='g', linestyle='-', label="Median")
ax_hist.axvline(mode, color='b', linestyle='-', label="Mode")
ax_hist.legend()
ax_box.set(xlabel='')
plt.show()
#previous
# Thiết lập màu nền (sử dụng sns.set_theme() nếu seaborn version 0.11.0 or above)
sns.set(rc={'figure.figsize': (11, 8)}, font_scale=1.5, style='whitegrid')
# Tạo figure kết hợp của 2 biểu đồ box và hist với matplotlib.Axes objects (ax_box and ax_hist)
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.3, 1)})

mean = df['previous'].mean()
median = df['previous'].median()
mode = df['previous'].mode().values[0]
sns.boxplot(data=df, x="previous", y="y", ax=ax_box, order=df["y"].value_counts().index)
ax_box.axvline(mean, color='r', linestyle='--')
ax_box.axvline(median, color='g', linestyle='-')
ax_box.axvline(mode, color='b', linestyle='-')

sns.histplot(data=df, x="previous", ax=ax_hist, kde=True)
ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
ax_hist.axvline(median, color='g', linestyle='-', label="Median")
ax_hist.axvline(mode, color='b', linestyle='-', label="Mode")
ax_hist.legend()
ax_box.set(xlabel='')
plt.show()
# Ma trận tương quan heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()
#==============================
#%% - Tạo cột mới, xóa cột cũ
new_df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'month', 'loan', 'contact', 'poutcome'], drop_first=True)
new_df.head()
# Chuyển đổi dữ liệu ( Chuyển categorical về 0 và 1)
new_df['y'].replace('yes', 1, inplace=True)
new_df['y'].replace('no', 0, inplace=True)
new_df.head()
# print(new_df.corr())
#%%
# df.drop(columns=["month", "previous", "day", "pdays"], inplace=True)
# print(df)
#%% - Thủ công
# df['default'] = df['default'].map({'yes': 1, 'no': 0})
# df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
# df['loan'] = df['loan'].map({'yes': 1, 'no': 0})
# df['contact'] = df['contact'].map({'telephone': 1, 'cellular': 0})
# df['y'] = df['y'].map({'yes': 1, 'no': 0})
# df
#%% - Mô hình đơn biến
# Xét biến độc lập 'housing', biến phụ thuộc 'y'
# Liệu rằng một người khi có khoản nợ mua nhà hay không có ảnh hưởng đến quyết định đăng ký tiền gửi của họ hay không?
# lấy ra biến housing và biến y trong tập dữ liệu
ndf = df[['housing', 'y']]
new_ndf = pd.get_dummies(ndf, columns=['housing'], drop_first=True)
new_ndf.head()
# Chuyển đổi dữ liệu ( Chuyển categorical về 0 và 1)
new_ndf['y'].replace('yes', 1, inplace=True)
new_ndf['y'].replace('no', 0, inplace=True)
new_ndf.head()
#%% - Chia dữ liệu train/test ( 75% train và 25% test)
uni_x = new_ndf.drop("y", axis='columns')
uni_y = new_ndf["y"]
x_train, x_test, y_train, y_test = train_test_split(uni_x, uni_y, random_state=0, train_size=.75)
#%% - Xây dựng mô hình hồi quy Logistic đơn biến
uni_model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
uni_model.fit(x_train, y_train)
y_uni_pred = uni_model.predict(x_test)
print(confusion_matrix(y_uni_pred, y_test))
print(uni_model.coef_, uni_model.intercept_)
print(classification_report(y_test, y_uni_pred))
# Ma trận nhầm lẫn
uni_cm = confusion_matrix(y_test, y_uni_pred)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(uni_cm, cmap=plt.cm.Blues, alpha=0.7)
for i in range(uni_cm.shape[0]):
    for j in range(uni_cm.shape[1]):
        ax.text(x=j, y=i, s=uni_cm[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
#=============================================================================
#%% - Mô hình đa biến
# - Chia dữ liệu train/test ( 75% train và 25% test)
x = df.drop("y", axis='columns')
y = df["y"]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0, train_size=.75)
#%% - Xây dựng model
# # Sử dụng statsmodels
# logit_model = sm.Logit(Y_train, X_train)
# result = logit_model.fit()
# print(result.summary())
lgt_model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
lgt_model.fit(X_train, Y_train)
y_pred = lgt_model.predict(X_test)
print(confusion_matrix(y_pred, y_test))
print(lgt_model.coef_, lgt_model.intercept_)
#In các thông số
print(confusion_matrix(Y_test, y_pred))
print(confusion_matrix(y_pred, Y_test))
print(classification_report(Y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
print("Precision:", metrics.precision_score(Y_test, y_pred))
print("Recall:", metrics.recall_score(Y_test, y_pred))
print('Cross Validation mean:', (cross_val_score(lgt_model, X_train, Y_train, cv=5, n_jobs=2, scoring='accuracy').mean()))
#%% - Ma trận nhầm lẫn
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