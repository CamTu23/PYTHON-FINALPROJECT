#%% - Import thư viện
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sms
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%% - Thiết kế thông số cơ bản cho biểu đồ
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 13
plt.rcParams["legend.loc"] = 'best'

#%% - Load data
df = pd.read_csv("./Data/insurance.csv")

#%% - Hiểu dữ liệu
df.info()
df.describe()
df.shape
# Tập dữ liệu có 1338 hàng và 7 cột.
# Nhìn vào dữ liệu, có vẻ như có một số trường có bản chất phân loại, nhưng ở kiểu số nguyên / float.
# Chúng ta sẽ phân tích và hoàn thiện xem nên chuyển chúng thành phân loại hay coi là số nguyên.

#%% - Kiểm tra phần trăm giá trị bị thiếu trong mỗi hàng
round(100*(df.isnull().sum()/len(df)),2).sort_values(ascending = False)
# Không có giá trị nào bị thiếu hoặc rỗng

#%% - Chuyển đổi sang kiểu dữ liệu 'category'
df['sex'] = df['sex'].astype('category')
df['smoker'] = df['smoker'].astype('category')
df['children'] = df['children'].astype('category')
df['region'] = df['region'].astype('category')
df_new = pd.get_dummies(df, drop_first=True)
df_new.info()
# Code trên thực hiện 3 bước sau
# 1) Tạo biến giả
# 2) Thả biến ban đầu mà giả đã được tạo
# 3) Thả biến giả đầu tiên cho mỗi nhóm mới được tạo.
df_new.shape
# Tập dữ liệu có 1338 hàng và 13 cột.

#%% - Tách dữ liệu để Huấn luyện và Kiểm tra:
# - Bây giờ chúng tôi sẽ chia dữ liệu thành ĐÀO TẠO và KIỂM TRA (tỷ lệ 80:20)
# Sử dụng phương thức train_test_split từ gói sklearn cho việc chia dữ liệu này
np.random.seed(0)
df_train, df_test = train_test_split(df_new, train_size = 0.80, test_size = 0.20, random_state = 100)
# Nên chỉ định 'random_state' để tập dữ liệu huấn luyện và thử nghiệm luôn có cùng các hàng, tương ứng
# Sau khi chia tập dữ liệu thì hiện tại df_test có 268 dòng và 13 cột. Còn df_train có 1070 dòng và 13 cột.

#%% - Thực hiện EDA
# Thực hiện EDA trên Tập dữ liệu TRAINING (df_train).
# Trực quan các biến số age và bmi
insurance_num = df_train[['age', 'bmi', 'charges']]
sns.pairplot(insurance_num, diag_kind='kde')
plt.show()
# Sơ đồ cặp ở trên cho chúng ta biết rằng có mối quan hệ tuyến tính giữa 'tuổi', 'bmi' và 'phí bảo hiểm'

#%% - Trực quan các biến số smoker và sex
insurance_num = df_train[['smoker_yes', 'sex_male', 'charges']]
sns.pairplot(insurance_num, diag_kind='kde')
plt.show()
# Sơ đồ cặp ở trên cho chúng ta biết rằng có mối quan hệ tuyến tính giữa 'người có hút thuốc' và 'phí bảo hiểm'. Còn giới tính và phí bảo hiểm không có mối quan hệ tuyến tính với nhau.

#%% - Trực quan biến số children
insurance_num = df_train[['children_1', 'children_2', 'children_3', 'children_4', 'children_5', 'charges']]
sns.pairplot(insurance_num, diag_kind='kde')
plt.show()
# Sơ đồ cặp ở trên cho chúng ta biết rằng có mối quan hệ tuyến tính giữa 'số trẻ em/người phụ thuộc' và 'phí bảo hiểm'.
#%% - Trực quan biến số region
insurance_num = df_train[['region_northwest', 'region_southeast', 'region_southwest', 'charges']]
sns.pairplot(insurance_num, diag_kind='kde')
plt.show()
# Sơ đồ cặp ở trên cho chúng ta biết rằng không có mối quan hệ tuyến tính giữa 'khu vực' và 'phí bảo hiểm'.
# Như vậy chỉ còn 4 biến độc lập là age, bmi, smoker và children là có sự tương quan với biến phụ thuộc charges.
#%% - Sơ đồ boxplot
plt.figure(figsize=(25, 10))
plt.subplot(2,2,1)
sns.boxplot(x = 'age', y = 'charges', data = df)
plt.subplot(2,2,2)
sns.boxplot(x = 'children', y = 'charges', data = df)
plt.subplot(2,2,3)
sns.boxplot(x = 'smoker', y = 'charges', data = df)
plt.subplot(2,2,4)
sns.boxplot(x = 'bmi', y = 'charges', data = df)
plt.show()
# Biến age: Những người có tuổi càng cao thì chi phí cũng càng cao.
# Biến children: Những người có 2 con đang có phân phối dữ liệu cao nhất từ 50 phần trăm đến 75 phần trăm trong số tất cả những người khác.
# Biến smoker: 20,5% dữ liệu là người hút thuốc. Mức phí trung bình của Người hút thuốc khá cao so với người không hút thuốc.
# Biến bmi: Những người có BMI càng cao thì chi phí cũng càng cao.

#%% - Trực quan bằng heatmap
plt.figure(figsize = (25,10))
sns.heatmap(df_new.corr(), annot = True, cmap="RdBu")
plt.show()
# Bản đồ nhiệt cho thấy rõ ràng tất cả các biến về bản chất là đa cộng tuyến và biến nào có tính cộng tuyến cao với biến mục tiêu.
# Chúng tôi sẽ tham khảo bản đồ này qua lại trong khi xây dựng mô hình tuyến tính để xác nhận các giá trị tương quan khác nhau cùng với VIF & p-value, nhằm xác định đúng biến để chọn / loại bỏ khỏi mô hình.
# Có sự tương quang giữa biến age và bmi. Kết luận là chọn 3 biến độc lập là bmi, children và smoker.

#%% -
scaler= MinMaxScaler()
df_train.head()
df_train.columns
num_vars = ['age', 'bmi', 'charges']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
df_train.describe()

#%% - Tạo model
y_train = df_train.pop('charges')
X_train = df_train

#%% - Loại bỏ tính năng đệ quy: Sử dụng chức năng LinearRegression từ SciKit Learn để tương thích với RFE (là một tiện ích từ sklearn)
# Chạy RFE với số đầu ra của biến bằng 6
lm= LinearRegression()
lm.fit(X_train,y_train)
rfe= RFE(lm,step=6)
rfe=rfe.fit(X_train,y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))

#%% - Biến phù hợp
col=X_train.columns[rfe.support_]
col
# Sau khi chạy lệnh thì cho ra các biến 'age', 'bmi', 'children_2', 'children_4', 'children_5', 'smoker_yes' là có tương quan với biến phụ thuộc charges.
#%% - Biến không phù hợp
X_train.columns[~rfe.support_]
# Sau khi chạy lệnh thì cho ra các biến 'sex_male', 'children_1', 'children_3', 'region_northwest', 'region_southeast', 'region_southwest' là không có tương quan với biến phụ thuộc charges.

#%% - Tạo model cho mô hình đa biến
X_train_rfe = X_train[col]
X_train_new = X_train_rfe.drop(["age", "children_4", "children_5"], axis = 1)
# Như vậy hiện tại chỉ còn ba biến độc lập là bmi, children_2 và smoker_yes. Loại bỏ age vì kết luận ở trên có sự tương quan với bmi. Còn loại bỏ biến children_4 và children_5 vì ở trên có kết luận biến children_2 là tương quan mạnh nhất.
#%%
# Tạo một khung dữ liệu sẽ chứa tên của tất cả các biến tính năng và các VIF tương ứng của chúng
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
#%% - Trả về kết quả
X_train_lm = sms.add_constant(X_train_new)
model = sms.OLS(y_train, X_train_lm).fit()
params = model.params
print(model.summary())
# Trả về kết quả là các giá trị hệ số là
# const 0.0266
#  bmi 0.2157
# children_2 0.0259
# smoker_yes 0.3740
# Và R-squared là 0.64
# charges = 0.0266 + 0.2157*bmi + 0.0259×children_2 + 0.3740*smoker_yes

# Chúng ta có thể thấy rằng phương trình của dây chuyền phù hợp nhất của chúng ta là:
# charges = 0.0266 + 0.3740*smoker_yes + 0.2157*bmi + 0.0259×children_2
#%% - Tiến hành dự báo cho model trên
num_vars = ['age', 'bmi', 'charges']
df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.head()
df_test.describe()

#%% - Chia y_test và X_test
y_test = df_test.pop('charges')
X_test = df_test
X_test.info()
#%%
col1=X_train_new.columns
X_test=X_test[col1]
X_test_lm = sms.add_constant(X_test)
X_test_lm.info()

#%% - Các giá trị dự báo
y_pred = model.predict(X_test_lm)

#%% - Trực quan hóa dữ liệu
plt.figure(figsize=(14,8))
sns.regplot(x=y_test, y=y_pred, ci=68, fit_reg=True,scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title('Predict y_test vs y_pred', fontsize=20)              # Plot heading
plt.xlabel('Actual', fontsize=18)                          # X-label
plt.ylabel('Predict', fontsize=16)                          # Y-label
plt.show()

#%% - Tạo model cho mô hình đơn biến somker_yes
X_train_rfe = X_train[col]
X_train_new = X_train_rfe.drop(["age", "children_4", "children_5", "children_2", "bmi"], axis = 1)
# Như vậy hiện tại chỉ còn ba biến độc lập là bmi, children_2 và smoker_yes. Loại bỏ age vì kết luận ở trên có sự tương quan với bmi. Còn loại bỏ biến children_4 và children_5 vì ở trên có kết luận biến children_2 là tương quan mạnh nhất.
#%%
# Tạo một khung dữ liệu sẽ chứa tên của tất cả các biến tính năng và các VIF tương ứng của chúng
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
#%% - Trả về kết quả
X_train_lm = sms.add_constant(X_train_new)
model1 = sms.OLS(y_train, X_train_lm).fit()
params = model.params
print(model1.summary())
# Trả về kết quả là các giá trị hệ số là
# const 0.1170
# smoker_yes 0.3734
# Và R-squared là 0.603
# charges = 0.1170 + 0.3734*smoker_yes



