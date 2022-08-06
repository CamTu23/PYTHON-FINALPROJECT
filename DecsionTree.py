#%% import
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#%%  thiết lập thông số cơ bnr cho biểu đồ
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] =150
plt.rcParams['font.size'] = 13

#%% nạp Dl phân tích
df=pd.read_csv('./Data/DL.csv')
x= df.drop('y',axis='columns')
y=df['y']

#%% processing data
from sklearn.preprocessing import LabelEncoder
# a={'<=30':0,'31..60':1,'>60':2}
# x['age_n'] = x['age'].map(a)
x['age_n']=LabelEncoder().fit_transform(x['age'])
x['job_n']=LabelEncoder().fit_transform(x['job'])
x['marital_n']=LabelEncoder().fit_transform(x['marital'])
x['education_n']=LabelEncoder().fit_transform(x['education'])
x['default_n']=LabelEncoder().fit_transform(x['default'])
x['balance_n']=LabelEncoder().fit_transform(x['balance'])
# b={'<=0':0,'1..50000':1,'>50000':2}
# x['balance_n'] = x['balance'].map(b)
x['housing_n']=LabelEncoder().fit_transform(x['housing'])
x['loan_n']=LabelEncoder().fit_transform(x['loan'])
x['contact_n']=LabelEncoder().fit_transform(x['contact'])
# d={'<=15':0,'>15':1}
# x['day_n'] = x['day'].map(d)
x['day_n']=LabelEncoder().fit_transform(x['day'])
x['month_n']=LabelEncoder().fit_transform(x['month'])
# dn={'<=1500 ':0,'1600..3000':1,'>3000':2}
# # x['duration_n'] = x['duration'].map(dn)
x['duration_n']=LabelEncoder().fit_transform(x['duration'])
# c={'<=20 ':0,'21..40':1,'>40':2}
# x['campaign_n'] = x['campaign'].map(c)
x['campaign_n']=LabelEncoder().fit_transform(x['campaign'])
# p={'<=365 ':0,'366..730':1,'>730':2}
# x['pdays_n'] = x['pdays'].map(p)
x['pdays_n']=LabelEncoder().fit_transform(x['pdays'])
# ps={'<=30 ':0,'31..60':1,'>60':2}
# x['previous_n'] = x['previous'].map(ps)
x['previous_n']=LabelEncoder().fit_transform(x['previous'])
x['poutcome_n']=LabelEncoder().fit_transform(x['poutcome'])

x_n=x.drop(['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'],axis='columns')
y_n=LabelEncoder().fit_transform(y)

#%% fit model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
#Supported criteria: Gini#Entropy:Information gain 100
model=DecisionTreeClassifier(criterion='gini',random_state=10,max_depth=3).fit(x_n, y_n)

#%%
score=model.score(x_n,y_n)

#%%
features=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
text_representation=tree.export_text(model,feature_names=features)
print(text_representation)

#%%
plt.figure(figsize=(20,20),dpi=150)
t=tree.plot_tree(model,feature_names=features,class_names=['No','Yes'],filled=True)
plt.show()

#%% Prediction





