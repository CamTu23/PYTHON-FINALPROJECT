#Decision Tree
#%% - Import thư viện
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
#%% - Thiết kế thông số cơ bản cho biểu đồ
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 13
plt.rcParams["legend.loc"] = 'best'

#%% - Load data
df = pd.read_csv("./Data/insurance.csv")

#%% Understanding data
df.head()
#%%
df.describe()

#%% Prepare data
x = df.drop('charges', axis=1)
y = df['charges']

#%%
x['age_n'] = LabelEncoder().fit_transform(x['age'])
x['bmi_n'] =LabelEncoder().fit_transform(x['bmi'])
x['children_n'] = LabelEncoder().fit_transform(x['children'])
x['sex_n'] = LabelEncoder().fit_transform(x['sex'])
x['smoker_n'] = LabelEncoder().fit_transform(x['smoker'])
x['region_n'] = LabelEncoder().fit_transform(x['region'])
x_n = x.drop(['age', 'bmi', 'children', 'sex', 'smoker', 'region'], axis = 'columns')
y_n = LabelEncoder().fit_transform(y)

#%% Split data
x_train, x_test, y_train, y_test = train_test_split(x_n, y_n, test_size=0.2, random_state=0)
#%% Create model and fit it
tree_insurance = DecisionTreeRegressor(max_depth=3)
tree_insurance.fit(x_train, y_train)

#%% - Visualize results
features = ['age', 'bmi', 'sex', 'children','smoker', 'region']
text_respresentation = tree.export_text(tree_insurance, feature_names= features)
print(text_respresentation)

#%%
plt.figure(figsize=(20,20), dpi = 150)
t = tree.plot_tree(tree_insurance, feature_names=features, filled=True)
plt.show()

#%%
y_pred = tree_insurance.predict(x_test)
#%% Evaluate
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# MAE < 10% MSE => tốt

#===============================================
# Random Forest
#%% Feature Scaling
sc = StandardScaler()
x_train_rf = sc.fit_transform(x_train)
x_test_rf = sc.transform(x_test)
#%% Create model and fit it
rf_insurance = RandomForestRegressor(n_estimators=20, random_state=0)
rf_insurance.fit(x_train_rf, y_train)
#%%Predict
y_pred = rf_insurance.predict(x_test_rf)

#%% Evaluate
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# => RMSE < 10% MSE => sử dụng tốt

#=====================================
#SVR
svr_insurance = SVR()
svr_insurance.fit(x_train, y_train)
#%% Predict
y_pred = svr_insurance.predict(x_test)
#%% Evaluate
print("R2-score:",metrics.r2_score(y_test, y_pred))
print("Mean squared log error:",metrics.mean_squared_log_error(y_test,y_pred))
#%% Visualation
plt.figure(figsize=(10,10))
sns.regplot(y_test, y_pred, fit_reg=True, scatter_kws={"s": 100})
plt.show()

