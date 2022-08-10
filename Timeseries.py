
#%% import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')

#%%
df=pd.read_csv('./Data/AirPassengers.csv',index_col='Month', parse_dates=True)
df.info()
plt.plot(df['#Passengers'])
plt.xlabel('Month')
plt.ylabel('Number Of Passengers')
plt.show()
#%%
print(df.isnull().values.any())
#%%
df_passengers=np.log(df['#Passengers'])
train_data,test_data=df_passengers[:int(len(df_passengers)*0.8)],df_passengers[int(len(df_passengers)*0.8):]
plt.plot(train_data,'blue',label='Train data')
plt.plot(test_data,'red',label='Test data')
plt.xlabel("Month")
plt.ylabel('Number Of Passengers')
plt.legend()
plt.show()

#%%
#Biểu đồ lịch sử so sánh số lượng hành khách với giá trị trung bình và độ lệch chuẩn : 12 kỳ trước
rolmean=train_data.rolling(12).mean()
rolstd=train_data.rolling(12).std()
plt.plot(train_data,'g',label='Original')
plt.plot(rolmean,'red',label='Rolling mean')
plt.plot(rolstd,'black',label='Rolling std')
plt.legend()
plt.show()
#Biểu đồ phân rã
decompose_results=seasonal_decompose(train_data,model='multiplicative', period=30)
decompose_results.plot()
plt.show()
#%%
#test = adfuller(train_data, autolag='AIC')
def adf_test(data):
    indices = ['ADF: Test statistic', 'p value', '# of Lags', '# of Observations']
    test = adfuller(data, autolag='AIC')
    results = pd.Series(test[:4], index=indices)
    for key, value in test[4].items():
        results[f'Critical Value({key})'] = value
    return results

print(adf_test(train_data))

##Nhận xét: Gỉa thiết ADF Ho là chuỗi không dừng. '. Có trị tuyệt đối ADF=1.57  nhỏ hơn trị tuyệt đối của tất các giá trị trong ngưỡng --> Không bác bỏ giả thiết Ho
##Kết luận: Ho là chuỗi không dừng

#%%
pd.plotting.lag_plot((train_data))
plt.show()
plot_pacf(train_data)
plt.show()
plot_acf(train_data)
plt.show()
## Nhận xét: Có sự tương quan cao

#%% Sai phân bậc 1
diff1=train_data.diff(1).dropna()
#trực quan
fig,ax=plt.subplots(2,sharex='all')
train_data.plot(ax=ax[0],title='Số hành khách ')
diff1.plot(ax=ax[1],title='Sai phân bậc nhất')
plt.show()
#kt lại
print(adf_test(diff1))

#%% Sai phân bậc 2
diff2=diff1.diff(1).dropna()
#trực quan
fig,ax=plt.subplots(2,sharex='all')
train_data.plot(ax=ax[0],title='Số hành khách ')
diff2.plot(ax=ax[1],title='Sai phân bậc hai')
plt.show()
#kt lại
print(adf_test(diff2))
# Giá trị ADF > giá trị critical => bác bỏ Ho => Dừng
#%%
# Xác định q,d,p tự tương quan
stepwise_fit=auto_arima(train_data,trace=True,suppress_warnings=True)
print(stepwise_fit.summary())
stepwise_fit.plot_diagnostics(figsize=(15,8))
plt.show()

#%%
# Áp dụng mô hình ARMA
modelARMA=ARIMA(train_data,order=(1,0,2),trend='t')
fitted=modelARMA.fit()
print(fitted.summary())

#%% Dự báo ARMA
predsARMA = fitted.get_forecast(len(test_data), alpha=0.05)
fc = predsARMA.predicted_mean
fc.index = test_data.index
conf = predsARMA.conf_int(alpha=0.05)
conf = conf.to_numpy()
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)

plt.figure(figsize=(10, 8), dpi=150)
plt.plot(train_data, label="Training data")
plt.plot(test_data, color="g", label="Actual ")
plt.plot(fc_series, color="r", label="Predicted ")
plt.fill_between(lower_series.index, lower_series, upper_series, color="b", alpha=.10)
plt.title('Passengers prediction with ARMA')
plt.xlabel("Month")
plt.ylabel("Number Of Passengers")
plt.legend()
plt.show()

#%%
# Áp dụng mô hình AR
modelAR=ARIMA(train_data,order=(1,0,0),trend='t')
fitted=modelAR.fit()
print(fitted.summary())

#%% Dự báo AR
predsAR= fitted.get_forecast(len(test_data), alpha=0.05)
fc = predsAR.predicted_mean
fc.index = test_data.index
conf = predsAR.conf_int(alpha=0.05)
conf = conf.to_numpy()
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)

plt.figure(figsize=(10, 8), dpi=150)
plt.plot(train_data, label="Training data")
plt.plot(test_data, color="g", label="Actual ")
plt.plot(fc_series, color="r", label="Predicted ")
plt.fill_between(lower_series.index, lower_series, upper_series, color="b", alpha=.10)
plt.title('Passengers prediction with AR')
plt.xlabel("Month")
plt.ylabel("Number Of Passengers")
plt.legend()
plt.show()

#%%
stepwise_fit_ARMIA=auto_arima(train_data,trace=True,suppress_warnings=True)
print(stepwise_fit_ARMIA.summary())
stepwise_fit_ARMIA.plot_diagnostics(figsize=(15,8))
plt.show()

#%% Áp dụng mô hình ARIMA
model_ARIMA=ARIMA(train_data,order=(5,1,2),trend='t')
fitted=model_ARIMA.fit()
print(fitted.summary())

#%% Dự báo ARIMA
predsARIMA = fitted.get_forecast(len(test_data), alpha=0.05)
fc = predsARIMA.predicted_mean
fc.index = test_data.index
conf = predsARIMA.conf_int(alpha=0.05)
conf = conf.to_numpy()
fc_series=pd.Series(fc,index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)

plt.figure(figsize=(10,8),dpi=150)
plt.plot(train_data,label="Training data")
plt.plot(test_data,color="g",label="Actual ")
plt.plot(fc_series,color="r",label="Predicted ")
plt.fill_between(lower_series.index,lower_series,upper_series,color="b",alpha=.10)
plt.title('Passengers prediction with ARIMA')
plt.xlabel("Month")
plt.ylabel("Number Of Passengers")
plt.legend()
plt.show()

#%% Evaluate for 3 model
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse,
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test_data.values)
print(forecast_accuracy(fc, test_data.values))

