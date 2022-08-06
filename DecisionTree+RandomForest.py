#%% import
from turtledemo import forest
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')
#%%  thiết lập thông số cơ bnr cho biểu đồ
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] =150
plt.rcParams['font.size'] = 13

#%% nạp Dl phân tích
df=pd.read_csv('./Data/DL.csv')
x = df.drop("y", axis='columns')
y = df["y"]

#%%processing data
from sklearn.preprocessing import LabelEncoder
x['age_n']=LabelEncoder().fit_transform(x['age'])
x['job_n']=LabelEncoder().fit_transform(x['job'])
x['marital_n']=LabelEncoder().fit_transform(x['marital'])
x['education_n']=LabelEncoder().fit_transform(x['education'])
x['default_n']=LabelEncoder().fit_transform(x['default'])
x['balance_n']=LabelEncoder().fit_transform(x['balance'])
x['housing_n']=LabelEncoder().fit_transform(x['housing'])
x['loan_n']=LabelEncoder().fit_transform(x['loan'])
x['contact_n']=LabelEncoder().fit_transform(x['contact'])
x['day_n']=LabelEncoder().fit_transform(x['day'])
x['month_n']=LabelEncoder().fit_transform(x['month'])
x['duration_n']=LabelEncoder().fit_transform(x['duration'])
x['campaign_n']=LabelEncoder().fit_transform(x['campaign'])
x['pdays_n']=LabelEncoder().fit_transform(x['pdays'])
x['previous_n']=LabelEncoder().fit_transform(x['previous'])
x['poutcome_n']=LabelEncoder().fit_transform(x['poutcome'])

x_n=x.drop(['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'],axis='columns')
y_n=LabelEncoder().fit_transform(y)

#%%
X_train, X_test, Y_train, Y_test = train_test_split(x_n, y_n, random_state=0, train_size=.75)

#%% Decision Tree
#% fit model
#Supported criteria: Gini#Entropy:Information gain 100
model=DecisionTreeClassifier(criterion='gini',random_state=10,max_depth=3).fit(X_train, Y_train)

#%%
score=model.score(X_train, Y_train)
y_pred_decision = model.predict(X_test)
#%%
features=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
text_representation=tree.export_text(model,feature_names=features)
print(text_representation)
print("Confusion matrix:\n",confusion_matrix(Y_test,y_pred_decision))
print("Classification report:\n",classification_report(Y_test,y_pred_decision))
print("Accuracy:",accuracy_score(Y_test, y_pred_decision))
#%%
plt.figure(figsize=(20,20),dpi=150)
t=tree.plot_tree(model,feature_names=features,class_names=['No','Yes'],filled=True)
plt.show()

#%% Randomforest
clf=RandomForestClassifier(n_estimators=100,max_depth=3).fit(X_train,Y_train)
#Train the model using the training sets y_pred=clf.predict(X_test)
y_pred_random=clf.predict(X_test)
#%%
#Import scikit-learn metrics module for accuracy calculation


print("Confusion matrix:\n",confusion_matrix(Y_test,y_pred_random))
print("Classification report:\n",classification_report(Y_test,y_pred_random))
print("Accuracy:",accuracy_score(Y_test, y_pred_random))



