#%% - Import thư viện
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

#%% - Thiết kế thông số cơ bản cho biểu đồ
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 13
# plt.rcParams['savefig.dpi'] = 200
# plt.rcParams['legend.fontsize'] = 'large'
# plt.rcParams['figure.titlesize'] = 'medium'
plt.rcParams["legend.loc"] = 'best'

#%% - Load data
df = pd.read_csv("./data/insurance.csv")

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
insurance_num = df_train[['children_1', 'children_2','children_3', 'children_4','children_5', 'charges']]
sns.pairplot(insurance_num, diag_kind='kde')
plt.show()
# Sơ đồ cặp ở trên cho chúng ta biết rằng có mối quan hệ tuyến tính giữa 'số trẻ em/người phụ thuộc' và 'phí bảo hiểm'.
#%% - Trực quan biến số region
insurance_num = df_train[['region_northwest', 'region_southeast', 'region_southwest', 'charges']]
sns.pairplot(insurance_num, diag_kind='kde')
plt.show()
# Sơ đồ cặp ở trên cho chúng ta biết rằng không có mối quan hệ tuyến tính giữa 'số trẻ em/người phụ thuộc' và 'phí bảo hiểm'.

#%% - Sơ đồ boxplot
# plt.figure(figsize=(25, 10))
# plt.subplot(2,2,1)
# sns.boxplot(x = 'sex', y = 'charges', data = df)
# plt.subplot(2,2,2)
# sns.boxplot(x = 'children', y = 'charges', data = df)
# plt.subplot(2,2,3)
# sns.boxplot(x = 'smoker', y = 'charges', data = df)
# plt.subplot(2,2,4)
# sns.boxplot(x = 'region', y = 'charges', data = df)
# plt.show()

#%% - Trực quan bằng heatmap
plt.figure(figsize = (25,10))
sns.heatmap(df_new.corr(), annot = True, cmap="RdBu")
plt.show()
# Bản đồ nhiệt cho thấy rõ ràng tất cả các biến về bản chất là đa cộng tuyến và biến nào có tính cộng tuyến cao với biến mục tiêu.
# Chúng tôi sẽ tham khảo bản đồ này qua lại trong khi xây dựng mô hình tuyến tính để xác nhận các giá trị tương quan khác nhau cùng với VIF & p-value, nhằm xác định đúng biến để chọn / loại bỏ khỏi mô hình.