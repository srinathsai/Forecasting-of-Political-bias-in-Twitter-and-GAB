#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import numpy as np


# In[2]:


df = pd.read_json('posts_with_newsURLs.json', lines=True, dtype=dict)


# In[3]:


df.head()


# In[4]:


df["data"][100]["created_at"]


# In[5]:


df_times=[]
df_body=[]


# In[6]:


for i in df.index:
    df_times.append(df["data"][i]["created_at"])


# In[7]:


print(df_times[12])


# In[8]:


for i in df.index:
    df_body.append(df["data"][i]["body"])


# In[9]:


print(df_body[112])


# In[10]:


urls=[]


# In[11]:


import re


# In[12]:


for i in df_body:
    m=str(i)
    s=re.search("(?P<url>https?://[^\s]+)", m).group(0)
    urls.append(s)


# In[13]:


print(len(urls))
print(len(df_times))


# In[14]:


h={'TimeStamps':df_times,'urls':urls}


# In[15]:


df_required=pd.DataFrame(h)


# In[16]:


df_required.dtypes


# In[17]:


df_required['TimeStamps'] = pd.to_datetime(df_required['TimeStamps'])


# In[18]:


df_required['urls'] = df_required['urls'].astype("string")


# In[19]:


df_required.head()


# In[20]:


df_required.info()


# In[21]:


#data preprocessing for dataset2


# In[22]:


data2=pd.DataFrame(pd.read_excel("MediaBias.xlsx"))


# In[23]:


data3 = data2[['Media source', 'URL', 'Political Leaning']].copy()


# In[24]:


#SPLITTING THE ENTIRE DATASET OF MEDIABIAS INTO LEFT,CENTRE AND RIGHT


# In[25]:


left_media=[]
right_media=[]
centre_media=[]


# In[26]:


data3.rename(columns = {'Media source':'Media_source', 'Political Leaning':'Political_Leaning'}, inplace = True)
data3.Media_source = data3.Media_source.astype('string')
data3.Political_Leaning=data3.Political_Leaning.astype('string')
data3.URL=data3.URL.astype('string')


# In[27]:


data3.dropna(inplace=True)


# In[28]:


for ind in data3.index:
     if((data3['Political_Leaning'][ind]=="Left") or (data3['Political_Leaning'][ind]=="Left Lean")):
       if(data3['Media_source'][ind] not in left_media):
        left_media.append(data3['Media_source'][ind])
     elif((data3['Political_Leaning'][ind]=="Right") or (data3['Political_Leaning'][ind]=="Right Lean" )):
       if(data3['Media_source'][ind] not in right_media):
        right_media.append(data3['Media_source'][ind])
     elif((data3['Political_Leaning'][ind]=="Center" ) or (data3['Political_Leaning'][ind]=="Mixed")):
       if(data3['Media_source'][ind] not in centre_media):
        centre_media.append(data3['Media_source'][ind])


# In[29]:


print(len(left_media))
print(len(right_media))
print(len(centre_media))


# In[30]:


hashmap_media={}


# In[31]:


for ind in data3.index:
    hashmap_media[data3['Media_source'][ind]]=data3['URL'][ind]


# In[32]:


left_timestamps=[]
right_timestamps=[]
centre_timestamps=[]


# In[33]:


import re


# In[34]:


df_required.isnull().values.any()


# In[35]:


left_suburls=[]
right_suburls=[]
centre_suburls=[]


# In[36]:


for i in range(0,len(left_media)):
    if hashmap_media.get(left_media[i]) not in left_suburls:
        left_suburls.append(hashmap_media.get(left_media[i]))


# In[37]:


print(len(left_suburls))
print((left_suburls))


# In[38]:


for i in range(0,len(right_media)):
    if hashmap_media.get(right_media[i]) not in right_suburls:
        right_suburls.append(hashmap_media.get(right_media[i]))


# In[39]:


print(len(right_suburls))
print(right_suburls)


# In[40]:


for i in range(0,len(centre_media)):
    if hashmap_media.get(centre_media[i]) not in centre_suburls:
        centre_suburls.append(hashmap_media.get(centre_media[i]))


# In[41]:


print(len(centre_suburls))


# In[42]:


left_timestamps=[]
right_timestamps=[]
centre_timestamps=[]


# In[43]:


x=df_required[["TimeStamps","urls"]].to_numpy()


# In[44]:


print(x)
print(len(x))
print(type(x[1]))
left=np.array(left_suburls)


# In[45]:


for y in x:
    for sub  in left_suburls:
        if y[1] is not np.nan:
            if sub in str(y[1]):
                left_timestamps.append(y[0].date())


# In[46]:


for y in x:
    for sub  in right_suburls:
        if y[1] is not np.nan:
            if sub in str(y[1]):
                right_timestamps.append(y[0].date())


# In[47]:


for y in x:
    for sub  in centre_suburls:
        if y[1] is not np.nan:
            if sub in str(y[1]):
                centre_timestamps.append(y[0].date())


# In[48]:


print(len(left_timestamps))
print(len(right_timestamps))
print(len(centre_timestamps))


# In[49]:


#Creating hashmap of frequency of tweets


# In[50]:


df_left_timestamps=pd.DataFrame(left_timestamps)


# In[51]:


df_left_timestamps['timestamps']=pd.DataFrame(left_timestamps)


# In[52]:


df_left_timestamps.head()


# In[53]:


df_right_timestamps=pd.DataFrame(right_timestamps)


# In[54]:


df_right_timestamps['timestamps']=pd.DataFrame(right_timestamps)


# In[55]:


df_right_timestamps.head()


# In[56]:


df_centre_timestamps=pd.DataFrame(centre_timestamps)


# In[57]:


df_centre_timestamps['timestamps']=pd.DataFrame(centre_timestamps)


# In[58]:


df_centre_timestamps.head()


# In[59]:


left_timestamp_frequency={}
right_timestamp_frequency={}
centre_timestamp_frequency={}


# In[60]:


for i in  df_left_timestamps.index :
    if df_left_timestamps['timestamps'][i] not in left_timestamp_frequency:
        left_timestamp_frequency[df_left_timestamps['timestamps'][i]]=1
    else:
        left_timestamp_frequency[df_left_timestamps['timestamps'][i]]+=1


# In[61]:


print(len(left_timestamp_frequency))


# In[62]:


for i in  df_right_timestamps.index :
    if df_right_timestamps['timestamps'][i] not in right_timestamp_frequency:
        right_timestamp_frequency[df_right_timestamps['timestamps'][i]]=1
    else:
        right_timestamp_frequency[df_right_timestamps['timestamps'][i]]+=1


# In[63]:


print(len(right_timestamp_frequency))


# In[64]:


for i in  df_centre_timestamps.index :
    if df_centre_timestamps['timestamps'][i] not in centre_timestamp_frequency:
        centre_timestamp_frequency[df_centre_timestamps['timestamps'][i]]=1
    else:
        centre_timestamp_frequency[df_centre_timestamps['timestamps'][i]]+=1


# In[65]:


print(len(centre_timestamp_frequency))


# In[66]:


#creating dataframes for models


# In[67]:


l1=[]
l2=[]


# In[68]:


for key in left_timestamp_frequency:
    l1.append(key)
    l2.append(left_timestamp_frequency[key])


# In[69]:


l = {'Date':l1,'frequency':l2}
left_dataset=pd.DataFrame(l)


# In[70]:


left_dataset.head()


# In[71]:


left_dataset['Date'] = pd.to_datetime(left_dataset['Date'], format='%Y-%m-%d')


# In[72]:


left_filtered=left_dataset.loc[(left_dataset['Date'] >= '2018-05-01') & (left_dataset['Date'] <= '2018-12-31')]


# In[73]:


left_filtered=left_filtered.drop_duplicates()


# In[74]:


print(left_filtered.size)


# In[75]:


left_filtered.head()


# In[76]:


r1=[]
r2=[]


# In[77]:


for key in right_timestamp_frequency:
    r1.append(key)
    r2.append(right_timestamp_frequency[key])


# In[78]:


r={'Date':r1,'frequency':r2}
right_dataset=pd.DataFrame(r)


# In[79]:


right_dataset.head()


# In[80]:


print(right_dataset.size)


# In[81]:


right_dataset['Date'] = pd.to_datetime(right_dataset['Date'], format='%Y-%m-%d')


# In[82]:


right_filtered=right_dataset.loc[(right_dataset['Date'] >= '2018-05-01') & (right_dataset['Date'] <='2018-12-31')]


# In[83]:


right_filtered=right_filtered.drop_duplicates()


# In[84]:


print(right_filtered.size)


# In[85]:


right_filtered.head()


# In[86]:


c1=[]
c2=[]


# In[87]:


for key in centre_timestamp_frequency:
    c1.append(key)
    c2.append(centre_timestamp_frequency[key])


# In[88]:


c={'Date':c1,'frequency':c2}
centre_dataset=pd.DataFrame(c)


# In[89]:


centre_dataset.head()


# In[90]:


print(centre_dataset.size)


# In[91]:


centre_dataset['Date'] = pd.to_datetime(centre_dataset['Date'], format='%Y-%m-%d')


# In[92]:


centre_filtered=centre_dataset.loc[(centre_dataset['Date'] >= '2018-05-01') & (centre_dataset['Date'] <= '2018-12-31')]


# In[93]:


centre_filtered=centre_filtered.drop_duplicates()


# In[94]:


print(centre_filtered.size)


# In[95]:


centre_filtered.head()


# In[96]:


#SARIMA MODEL FOR LEFT TWEETS 


# In[97]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[98]:


from datetime import datetime


# In[99]:


left_filtered['Date']=pd.to_datetime(left_filtered['Date'])
left_filtered.set_index('Date', inplace=True)
#check datatype of index
left_filtered.index      


# In[100]:


left_ts = left_filtered['frequency']
left_ts.head(10)


# In[101]:


left_ts=left_ts.sort_index(ascending=True)


# In[102]:


from statsmodels.tsa.stattools import adfuller
def check_stationarity(ts):
    dftest = adfuller(ts)
    adf = dftest[0]
    pvalue = dftest[1]
    critical_value = dftest[4]['5%']
    if (pvalue < 0.05) and (adf < critical_value):      #here we are defining a function that uses Dicky fuller method and prints the results based on pvalue and test static value
        print('The series is stationary')
    else:
        print('The series is NOT stationary')


# In[103]:


check_stationarity(left_ts)


# In[104]:


plt.plot(left_ts)


# In[105]:


from IPython.display import display

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from time import time
import seaborn as sns
sns.set(style="whitegrid")

import warnings


# In[106]:


result = seasonal_decompose(left_ts,model='additive',period=15)      #applied seasonal decomposition to get trend and seasons.
fig = result.plot()


# In[107]:


plot_acf(left_ts, lags=20);
plot_pacf(left_ts, lags=20);  #getting order of Auto regression (p) and getting number of forecast errors(q) to specify for the model.
                         #p is point where acf increased significantly, it can be any points but in our graph we get increasing trend at 0.


# In[108]:


sarima = SARIMAX(left_ts, order=(7,1,10), seasonal_order=(3,1,1,14))
sarima_predictions = sarima.fit().predict()                            #Links to get SARIMAX parameters :- https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
                                              


# In[109]:


plt.figure(figsize=(16,4))
plt.plot(left_ts, label="Actual")
plt.plot(sarima_predictions, label="Predicted")
plt.title('frequency of tweets', fontsize=20)
#plt.ylabel('Sales', fontsize=16)
plt.legend()


# In[110]:


print('RMSE OF SARIMA OF LEFT TWEETS: %.4f'%np.sqrt(sum((left_ts-sarima_predictions)**2)/(len(left_ts))))


# In[111]:


#SARIMA MODEL FOR RIGHT TWEETS PREDICTION


# In[112]:


right_filtered['Date']=pd.to_datetime(right_filtered['Date'])
right_filtered.set_index('Date', inplace=True)
#check datatype of index
right_filtered.index      


# In[113]:


right_ts = right_filtered['frequency']
right_ts.head(10)


# In[114]:


right_ts=right_ts.sort_index(ascending=True)


# In[115]:


check_stationarity(right_ts)


# In[116]:


plt.plot(right_ts)


# In[117]:


result = seasonal_decompose(right_ts,model='additive',period=10)      #applied seasonal decomposition to get trend and seasons.
fig = result.plot()


# In[118]:


plot_acf(right_ts, lags=20);
plot_pacf(right_ts, lags=20);  


# In[119]:


sarima = SARIMAX(right_ts, order=(6,2,10), seasonal_order=(4,1,1,11))
sarima_predictions = sarima.fit().predict()            


# In[120]:


plt.figure(figsize=(16,4))
plt.plot(right_ts, label="Actual")
plt.plot(sarima_predictions, label="Predicted")
plt.title('frequency of tweets', fontsize=20)
#plt.ylabel('Sales', fontsize=16)
plt.legend()


# In[121]:


print('RMSE OF SARIMA OF RIGHT TWEETS: %.4f'%np.sqrt(sum((right_ts-sarima_predictions)**2)/(len(right_ts))))


# In[122]:


#SARIMA model for centre tweets


# In[123]:


centre_filtered['Date']=pd.to_datetime(centre_filtered['Date'])
centre_filtered.set_index('Date', inplace=True)
#check datatype of index
centre_filtered.index    


# In[124]:


centre_ts = centre_filtered['frequency']
centre_ts.head(10)


# In[125]:


centre_ts=centre_ts.sort_index(ascending=True)


# In[126]:


check_stationarity(centre_ts)


# In[127]:


plt.plot(centre_ts)


# In[128]:


result = seasonal_decompose(centre_ts,model='additive',period=10)      #applied seasonal decomposition to get trend and seasons.
fig = result.plot()


# In[129]:


plot_acf(centre_ts, lags=20);
plot_pacf(centre_ts, lags=20);  


# In[130]:


sarima = SARIMAX(centre_ts, order=(11,1,10), seasonal_order=(2,1,1,14))
sarima_predictions = sarima.fit().predict()    


# In[131]:


plt.figure(figsize=(16,4))
plt.plot(centre_ts, label="Actual")
plt.plot(sarima_predictions, label="Predicted")
plt.title('frequency of tweets', fontsize=20)
#plt.ylabel('Sales', fontsize=16)
plt.legend()


# In[132]:


print('RMSE OF SARIMA OF CENTRE TWEETS: %.4f'%np.sqrt(sum((centre_ts-sarima_predictions)**2)/(len(centre_ts))))


# In[133]:


import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import read_csv


# In[134]:


#GRID SEARCH FOR HYPER PARAMETERS OF SARIMA MODEL


# In[135]:


from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
 
# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]
 
# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error
 
# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)
 
# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores
 
# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n','c','t','ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
    return models


# In[136]:


#LSTM MODEL FOR LEFT TWEETS PREDICTION FOR FEEDBACK DAY =1


# In[137]:


dataset=left_ts


# In[138]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1,1))
train_size = int(len(dataset) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[139]:


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):   #The function takes two arguments: the dataset, which is a NumPy array that we want to convert into a dataset, 
  #and the look_back, which is the number of previous time steps to use as input variables to predict the next time period — in this case defaulted to 1.
#This default will create a dataset where X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[140]:


#hyper parameter tunning for Left tweets LSTM model


# In[141]:


import tensorflow as f
from tensorflow import keras
import keras_tuner as kt


# In[142]:


import numpy as np
from keras.layers import LSTM,Input
from keras.models import Sequential


# In[143]:


from tensorflow import keras
from keras_tuner.tuners import BayesianOptimization
import os
n_input = 1
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units',min_value=32,
                                    max_value=512,
                                    step=32), 
               activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(units=hp.Int('units',min_value=32,
                                    max_value=512,
                                    step=32), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', metrics=['mse'], optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate',
                  values=[1e-2, 1e-3, 1e-4])))

    return model

bayesian_opt_tuner = BayesianOptimization(
    build_model,
    objective='mse',
    max_trials=2, 
    executions_per_trial=1,
    directory=os.path.normpath('C:/keras_tuning'),
    project_name='kerastuner_bayesian_poc',
    overwrite=True)

bayesian_opt_tuner.search(train, train,epochs=1,
     #validation_data=(X_test, y_test)
     validation_split=0.2,verbose=1)


bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)
model = bayes_opt_model_best_model[0]


# In[144]:


#FITTING TO THE MODEL


# In[145]:


look_back = 1
trainX, trainY = create_dataset(train, look_back) #converting as above mentioned function for training and testing.
testX, testY = create_dataset(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))    


# In[146]:


model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back))) #4 are the hidden LSTM blocks and 1 input layer as visible.
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='RMSProp')
model.fit(trainX, trainY, epochs=200, batch_size=4, verbose=2) #training the data with epochs=100,batch size=1;


# In[147]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[148]:


model.summary()


# In[149]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[150]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score RMSE OF LEFT TWEETS: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score RMSE OF LEFT TWEETS: %.2f RMSE' % (testScore))


# In[151]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)         
trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[152]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
plt.show()


# In[153]:


#centre tweets prediction by LSTM MODEL for lookback=1


# In[154]:


dataset=centre_ts


# In[155]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1,1))


# In[156]:


train_size = int(len(dataset) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[157]:


#centre tweets hyperparameters tunning


# In[158]:


from tensorflow import keras
from keras_tuner.tuners import BayesianOptimization
import os
n_input = 1
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units',min_value=32,
                                    max_value=512,
                                    step=32), 
               activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(units=hp.Int('units',min_value=32,
                                    max_value=512,
                                    step=32), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', metrics=['mse'], optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate',
                  values=[1e-2, 1e-3, 1e-4])))

    return model

bayesian_opt_tuner = BayesianOptimization(
    build_model,
    objective='mse',
    max_trials=3,
    executions_per_trial=1,
    directory=os.path.normpath('C:/keras_tuning'),
    project_name='kerastuner_bayesian_poc',
    overwrite=True)

bayesian_opt_tuner.search(train, train,epochs=100,
     #validation_data=(X_test, y_test)
     validation_split=0.2,verbose=1)


bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)
model = bayes_opt_model_best_model[0]


# In[159]:


look_back = 1
trainX, trainY = create_dataset(train, look_back) #converting as above mentioned function for training and testing.
testX, testY = create_dataset(test, look_back)


# In[160]:


trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[161]:


model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back))) #4 are the hidden LSTM blocks and 1 input layer as visible.
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='RMSProp')
model.fit(trainX, trainY, epochs=200, batch_size=4, verbose=2) #training the data with epochs=100,batch size=1;


# In[162]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[163]:


model.summary()


# In[164]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[165]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score RMSE OF centre tweets: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score RMSE OF centre tweets: %.2f RMSE' % (testScore))


# In[166]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)         
trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[167]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
plt.show()


# In[168]:


#LSTM FOR RIGHT TWEETS FOR FEEDBACK=1


# In[169]:


dataset=right_ts


# In[170]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1,1))    


# In[171]:


train_size = int(len(dataset) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[172]:


look_back = 1
trainX, trainY = create_dataset(train, look_back) #converting as above mentioned function for training and testing.
testX, testY = create_dataset(test, look_back)


# In[173]:


trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))    


# In[174]:


model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back))) #4 are the hidden LSTM blocks and 1 input layer as visible.
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=2, verbose=2) #training the data with epochs=100,batch size=1;


# In[175]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[176]:


model.summary()


# In[177]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[178]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score RMSE OF right tweets: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score RMSE OF right tweets: %.2f RMSE' % (testScore))


# In[179]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)         
trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[180]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
plt.show()


# In[181]:


#left tweets prediction using multiple lookbacks.


# In[182]:


dataset1=left_ts


# In[183]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(dataset1.values.reshape(-1,1))
train_size = int(len(dataset1) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset1) - train_size
train, test = dataset1[0:train_size,:], dataset1[train_size:len(dataset1),:]
print(len(train), len(test))


# In[184]:


for i in range(3,4):
    trainX, trainY = create_dataset(train, i) #converting as above mentioned function for training and testing.
    testX, testY = create_dataset(test, i)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, i))) #4 are the hidden LSTM blocks and 1 input layer as visible.
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='RMSProp')
    model.fit(trainX, trainY, epochs=200, batch_size=4, verbose=2) #training the data with epochs=100,batch size=1;
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score RMSE OF left tweets: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score RMSE OF left tweets: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset1)         
    trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
    trainPredictPlot[i:len(trainPredict)+i, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(i*2)+1:len(dataset1)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset1))
    plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
    plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
    plt.show()
    model.summary()


# In[185]:


#centre tweets prediction for multiple lookbacks


# In[186]:


dataset2=centre_ts


# In[187]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset2 = scaler.fit_transform(dataset2.values.reshape(-1,1))
train_size = int(len(dataset1) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset2) - train_size
train, test = dataset2[0:train_size,:], dataset2[train_size:len(dataset2),:]
print(len(train), len(test))


# In[188]:


for i in range(3,4):
    trainX, trainY = create_dataset(train, i) #converting as above mentioned function for training and testing.
    testX, testY = create_dataset(test, i)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, i))) #4 are the hidden LSTM blocks and 1 input layer as visible.
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='RMSProp')
    model.fit(trainX, trainY, epochs=200, batch_size=4, verbose=2) #training the data with epochs=100,batch size=1;
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score RMSE OF centre tweets: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score RMSE OF centre tweets: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset2)         
    trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
    trainPredictPlot[i:len(trainPredict)+i, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset2)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(i*2)+1:len(dataset2)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset2))
    plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
    plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
    plt.show()
    model.summary()


# In[189]:


# right tweets prediction for multiple lookbacks


# In[190]:


dataset3=right_ts


# In[191]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset3 = scaler.fit_transform(dataset3.values.reshape(-1,1))
train_size = int(len(dataset3) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset3) - train_size
train, test = dataset3[0:train_size,:], dataset3[train_size:len(dataset2),:]
print(len(train), len(test))


# In[192]:


for i in range(3,4):
    trainX, trainY = create_dataset(train, i) #converting as above mentioned function for training and testing.
    testX, testY = create_dataset(test, i)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, i))) #4 are the hidden LSTM blocks and 1 input layer as visible.
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=2, verbose=2) #training the data with epochs=100,batch size=1;
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score RMSE OF right tweets: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score RMSE OF right tweets: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset3)         
    trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
    trainPredictPlot[i:len(trainPredict)+i, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset3)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(i*2)+1:len(dataset3)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset3))
    plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
    plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
    plt.show()
    model.summary()


# In[193]:


# multistep time series of lstm


# In[194]:


from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array


# In[195]:


def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
 
# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
 
# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test
# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='RMSProp')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2, shuffle=False)
        model.reset_states()
    return model
 
# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]
 
# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts
# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted
 
# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted
# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))
 
# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    pyplot.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
    # show the plot
    pyplot.show()
 


# In[196]:


#Multistep time series forecasting for left tweets


# In[197]:


# load dataset
series = left_ts
# configure
n_lag = 3
n_seq = 6
n_test = 37
n_epochs = 200
n_batch = 1
n_neurons = 1
# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(series, forecasts, n_test+2)


# In[198]:


model.summary()


# In[199]:


#right multistep time series of lstm


# In[200]:


# load dataset
series = right_ts
# configure
n_lag = 3
n_seq = 6
n_test = 37
n_epochs = 75
n_batch = 1
n_neurons = 1
# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(series, forecasts, n_test+2)


# In[201]:


model.summary()


# In[202]:


# load dataset
series = centre_ts
# configure
n_lag = 3
n_seq = 6
n_test = 37
n_epochs = 150
n_batch = 1
n_neurons = 1
# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(series, forecasts, n_test+2)


# In[203]:


model.summary()


# In[204]:


# preparing datset for adaption rate


# In[205]:


#like_count
#repost_count
#dislike_count


# In[206]:


print(df["data"][0]["like_count"])
print(df["data"][0]["repost_count"])
print(df["data"][0]["dislike_count"])


# In[207]:


likes=[]
dislikes=[]
repost=[]


# In[208]:


for i in df.index:
    likes.append(df["data"][i]["like_count"])
    dislikes.append(df["data"][i]["dislike_count"])
    repost.append(df["data"][i]["repost_count"])


# In[209]:


df_required["likes_count"]=likes
df_required["dislikes"]=dislikes
df_required["repost"]=repost


# In[210]:


df_required.info()


# In[211]:


x1=df_required[["TimeStamps","urls","likes_count","dislikes","repost"]].to_numpy()


# In[212]:


left_timestamps1=[]
right_timestamps1=[]
centre_timestamps1=[]
left_likes=[]
right_likes=[]
centre_likes=[]
left_dislikes=[]
centre_dislikes=[]
right_dislikes=[]
left_reposts=[]
right_reposts=[]
centre_reposts=[]


# In[213]:


#getting left like counts


# In[214]:


for y in x1:
    for sub  in left_suburls:
        if y[1] is not np.nan:
            if sub in str(y[1]):
                left_timestamps1.append(y[0].date())
                left_likes.append(y[2])
                left_dislikes.append(y[3])
                left_reposts.append(y[4])


# In[215]:


df_left_timestamps1=pd.DataFrame()


# In[216]:


df_left_timestamps1['timestamps']=pd.DataFrame(left_timestamps1)


# In[217]:


df_left_timestamps1['likes']=left_likes


# In[218]:


df_left_timestamps1.head()


# In[219]:


df_left_timestamps1.info()


# In[220]:


df_left_timestamps2=df_left_timestamps1.drop_duplicates(keep='first')


# In[221]:


df_left_timestamps2.info()


# In[222]:


left_likes_frequency=dict(zip(df_left_timestamps2.timestamps, df_left_timestamps2.likes))


# In[223]:


for i in range(6476,len(df_left_timestamps1)):
    left_likes_frequency[df_left_timestamps1.loc[i,"timestamps"]]+=df_left_timestamps1.loc[i,"likes"]


# In[224]:


print(len(left_likes_frequency))


# In[225]:


t1=[]
like=[]


# In[226]:


for key in left_likes_frequency:
    t1.append(key)
    like.append(left_likes_frequency[key])


# In[227]:


l3 = {'Date':t1,'likes_count':like}
left_likes_dataset=pd.DataFrame(l3)


# In[228]:


left_likes_dataset.head()


# In[229]:


#getting left dislike counts


# In[230]:


df_left_timestamps4=pd.DataFrame()


# In[231]:


df_left_timestamps4['timestamps']=pd.DataFrame(left_timestamps1)


# In[232]:


df_left_timestamps4['dislikes']=left_dislikes


# In[233]:


df_left_timestamps4.head()


# In[234]:


df_left_timestamps4.info()


# In[235]:


df_left_timestamps5=df_left_timestamps4.drop_duplicates(keep='first')


# In[236]:


df_left_timestamps5.info()


# In[237]:


left_dislikes_frequency=dict(zip(df_left_timestamps5.timestamps, df_left_timestamps5.dislikes))


# In[238]:


for i in range(872,len(df_left_timestamps4)):
    left_dislikes_frequency[df_left_timestamps4.loc[i,"timestamps"]]+=df_left_timestamps4.loc[i,"dislikes"]


# In[239]:


t2=[]
dislike=[]


# In[240]:


for key in left_dislikes_frequency:
    t2.append(key)
    dislike.append(left_dislikes_frequency[key])


# In[241]:


l4 = {'Date':t2,'dislikes_count':dislike}
left_dislikes_dataset=pd.DataFrame(l4)


# In[242]:


left_dislikes_dataset.head()


# In[243]:


#getting left repost counts


# In[244]:


df_left_timestamps6=pd.DataFrame()


# In[245]:


df_left_timestamps6['timestamps']=pd.DataFrame(left_timestamps1)


# In[246]:


df_left_timestamps6['reposts']=left_reposts


# In[247]:


df_left_timestamps6.info()


# In[248]:


df_left_timestamps7=df_left_timestamps6.drop_duplicates(keep='first')


# In[249]:


df_left_timestamps7.info()


# In[250]:


left_reposts_frequency=dict(zip(df_left_timestamps7.timestamps, df_left_timestamps7.reposts))


# In[251]:


for i in range(3839,len(df_left_timestamps6)):
    left_reposts_frequency[df_left_timestamps6.loc[i,"timestamps"]]+=df_left_timestamps6.loc[i,"reposts"]


# In[252]:


def cleanNullTerms(d):
   return {
      k:v
      for k, v in d.items()
      if v is not None
   }


# In[253]:


t3=[]
repost=[]


# In[254]:


for key in left_reposts_frequency:
    t3.append(key)
    repost.append(left_reposts_frequency[key])


# In[255]:


l5= {'Date':t3,'reposts_count':repost}
left_reposts_dataset=pd.DataFrame(l5)


# In[256]:


left_reposts_dataset.head()


# In[257]:


#combining all the counts 


# In[258]:


left_df_adoption=pd.DataFrame()


# In[259]:


left_df_adoption["date"]=left_reposts_dataset["Date"]


# In[260]:


left_df_adoption["LIKES_COUNT"]=left_likes_dataset["likes_count"]
left_df_adoption["DISLIKES_COUNT"]=left_dislikes_dataset["dislikes_count"]
left_df_adoption["REPOSTS_COUNT"]=left_reposts_dataset["reposts_count"]


# In[261]:


left_df_adoption.info()


# In[262]:


left_df_adoption.head()


# In[263]:


#getting right tweets like counts


# In[264]:


for y in x1:
    for sub  in right_suburls:
        if y[1] is not np.nan:
            if sub in str(y[1]):
                right_timestamps1.append(y[0].date())
                right_likes.append(y[2])
                right_dislikes.append(y[3])
                right_reposts.append(y[4])


# In[265]:


df_right_timestamps1=pd.DataFrame()


# In[266]:


df_right_timestamps1['timestamps']=pd.DataFrame(right_timestamps1)


# In[267]:


df_right_timestamps1['likes']=right_likes


# In[268]:


df_right_timestamps1.info()


# In[269]:


df_right_timestamps1.head()


# In[270]:


df_right_timestamps2=df_right_timestamps1.drop_duplicates(keep='first')


# In[271]:


df_right_timestamps2.info()


# In[272]:


right_likes_frequency=dict(zip(df_right_timestamps2.timestamps, df_right_timestamps2.likes))


# In[273]:


for i in range(12704,len(df_right_timestamps1)):
    right_likes_frequency[df_right_timestamps1.loc[i,"timestamps"]]+=df_right_timestamps1.loc[i,"likes"]


# In[274]:


r1=[]
rlike=[]


# In[275]:


for key in right_likes_frequency:
    r1.append(key)
    rlike.append(right_likes_frequency[key])


# In[276]:


l3 = {'Date':r1,'likes_count':rlike}
right_likes_dataset=pd.DataFrame(l3)


# In[277]:


right_likes_dataset.head()


# In[278]:


#getting right tweets dislike counts


# In[279]:


df_right_timestamps3=pd.DataFrame()


# In[280]:


df_right_timestamps3['timestamps']=pd.DataFrame(right_timestamps1)


# In[281]:


df_right_timestamps3['dislikes']=right_dislikes


# In[282]:


df_right_timestamps3.head()


# In[283]:


df_right_timestamps4=df_right_timestamps3.drop_duplicates(keep='first')


# In[284]:


df_right_timestamps4.info()


# In[285]:


right_dislikes_frequency=dict(zip(df_right_timestamps4.timestamps, df_right_timestamps4.dislikes))


# In[286]:


for i in range(936,len(df_right_timestamps3)):
    right_dislikes_frequency[df_right_timestamps3.loc[i,"timestamps"]]+=df_right_timestamps3.loc[i,"dislikes"]


# In[287]:


r2=[]
rdislike=[]


# In[288]:


for key in right_dislikes_frequency:
    r2.append(key)
    rdislike.append(right_dislikes_frequency[key])


# In[289]:


l4= {'Date':r2,'dislikes_count':rdislike}
right_dislikes_dataset=pd.DataFrame(l4)


# In[290]:


right_dislikes_dataset.head()


# In[291]:


#getting right tweet reposts counts


# In[292]:


df_right_timestamps5=pd.DataFrame()


# In[293]:


df_right_timestamps5['timestamps']=pd.DataFrame(right_timestamps1)


# In[294]:


df_right_timestamps5['reposts']=right_reposts


# In[295]:


df_right_timestamps5.head()


# In[296]:


df_right_timestamps6=df_right_timestamps5.drop_duplicates(keep='first')


# In[297]:


df_right_timestamps6.info()


# In[298]:


right_reposts_frequency=dict(zip(df_right_timestamps6.timestamps, df_right_timestamps6.reposts))


# In[299]:


for i in range(8233,len(df_right_timestamps5)):
    right_reposts_frequency[df_right_timestamps5.loc[i,"timestamps"]]+=df_right_timestamps5.loc[i,"reposts"]


# In[300]:


r3=[]
rreposts=[]


# In[301]:


for key in right_reposts_frequency:
    r3.append(key)
    rreposts.append(right_reposts_frequency[key])


# In[302]:


l5= {'Date':r3,'reposts_count':rreposts}
right_reposts_dataset=pd.DataFrame(l5)


# In[303]:


right_reposts_dataset.head()


# In[304]:


#combining all in a single dataframe


# In[305]:


right_df_adoption=pd.DataFrame()


# In[306]:


right_df_adoption["date"]=right_likes_dataset["Date"]
right_df_adoption["LIKES_COUNT"]=right_likes_dataset["likes_count"]
right_df_adoption["DISLIKES_COUNT"]=right_dislikes_dataset["dislikes_count"]
right_df_adoption["REPOSTS_COUNT"]=right_reposts_dataset["reposts_count"]


# In[307]:


right_df_adoption.head()


# In[308]:


#preparing adoption rate dataset for centre 


# In[309]:


#getting centre like counts


# In[310]:


for y in x1:
    for sub  in centre_suburls:
        if y[1] is not np.nan:
            if sub in str(y[1]):
                centre_timestamps1.append(y[0].date())
                centre_likes.append(y[2])
                centre_dislikes.append(y[3])
                centre_reposts.append(y[4])


# In[311]:


df_centre_timestamps1=pd.DataFrame()


# In[312]:


df_centre_timestamps1['timestamps']=pd.DataFrame(centre_timestamps1)
df_centre_timestamps1['likes']=centre_likes


# In[313]:


df_centre_timestamps1.head()


# In[314]:


df_centre_timestamps2=df_centre_timestamps1.drop_duplicates(keep='first')


# In[315]:


df_centre_timestamps2.info()


# In[316]:


centre_likes_frequency=dict(zip(df_centre_timestamps2.timestamps, df_centre_timestamps2.likes))


# In[317]:


for i in range(5856,len(df_centre_timestamps1)):
    centre_likes_frequency[df_centre_timestamps1.loc[i,"timestamps"]]+=df_centre_timestamps1.loc[i,"likes"]


# In[318]:


c1=[]
clike=[]


# In[319]:


for key in centre_likes_frequency:
    c1.append(key)
    clike.append(centre_likes_frequency[key])


# In[320]:


c3 = {'Date':c1,'likes_count':clike}
centre_likes_dataset=pd.DataFrame(c3)


# In[321]:


centre_likes_dataset.head()


# In[322]:


#getting centre dislikes count


# In[323]:


df_centre_timestamps3=pd.DataFrame()


# In[324]:


df_centre_timestamps3['timestamps']=pd.DataFrame(centre_timestamps1)
df_centre_timestamps3['dislikes']=centre_dislikes


# In[325]:


df_centre_timestamps3.head()


# In[326]:


df_centre_timestamps4=df_centre_timestamps3.drop_duplicates(keep='first')


# In[327]:


df_centre_timestamps4.info()


# In[328]:


centre_dislikes_frequency=dict(zip(df_centre_timestamps4.timestamps, df_centre_timestamps4.dislikes))


# In[329]:


for i in range(733,len(df_centre_timestamps3)):
    centre_dislikes_frequency[df_centre_timestamps3.loc[i,"timestamps"]]+=df_centre_timestamps3.loc[i,"dislikes"]


# In[330]:


c2=[]
cdislike=[]


# In[331]:


for key in centre_dislikes_frequency:
    c2.append(key)
    cdislike.append(centre_dislikes_frequency[key])


# In[332]:


c4 = {'Date':c2,'dislikes_count':cdislike}
centre_dislikes_dataset=pd.DataFrame(c4)


# In[333]:


centre_dislikes_dataset.head()


# In[334]:


#getting centre reposts count


# In[335]:


df_centre_timestamps5=pd.DataFrame()


# In[336]:


df_centre_timestamps5['timestamps']=pd.DataFrame(centre_timestamps1)
df_centre_timestamps5['reposts']=centre_reposts


# In[337]:


df_centre_timestamps5.head()


# In[338]:


df_centre_timestamps6=df_centre_timestamps5.drop_duplicates(keep='first')


# In[339]:


df_centre_timestamps6.info()


# In[340]:


centre_reposts_frequency=dict(zip(df_centre_timestamps6.timestamps, df_centre_timestamps6.reposts))


# In[341]:


for i in range(3610,len(df_centre_timestamps5)):
    centre_reposts_frequency[df_centre_timestamps5.loc[i,"timestamps"]]+=df_centre_timestamps5.loc[i,"reposts"]


# In[342]:


c3=[]
creposts=[]


# In[343]:


for key in centre_reposts_frequency:
    c3.append(key)
    creposts.append(centre_reposts_frequency[key])


# In[344]:


c5 = {'Date':c3,'reposts_count':creposts}
centre_reposts_dataset=pd.DataFrame(c5)


# In[345]:


centre_reposts_dataset.head()


# In[346]:


#combining all centre counts


# In[347]:


centre_df_adoption=pd.DataFrame()


# In[348]:


centre_df_adoption["date"]=centre_likes_dataset["Date"]
centre_df_adoption["LIKES_COUNT"]=centre_likes_dataset["likes_count"]
centre_df_adoption["DISLIKES_COUNT"]=centre_dislikes_dataset["dislikes_count"]
centre_df_adoption["REPOSTS_COUNT"]=centre_reposts_dataset["reposts_count"]


# In[349]:


centre_df_adoption.head()


# In[350]:


right_df_adoption.head()


# In[351]:


left_df_adoption.head()


# In[352]:


#LIKES COUNT FORECASTING


# In[353]:


left_likes_dataset.info()


# In[354]:


# likes forecasting for left leaning dataset SARIMA


# In[355]:


left_likes_dataset['Date'] = pd.to_datetime(left_dataset['Date'], format='%Y-%m-%d')


# In[356]:


left_likes_filtered=left_likes_dataset.loc[(left_likes_dataset['Date'] >= '2018-05-01') & (left_likes_dataset['Date'] <= '2018-12-31')]


# In[357]:


from datetime import datetime
con=left_likes_filtered['Date']
left_likes_filtered['Date']=pd.to_datetime(left_likes_filtered['Date'])
left_likes_filtered.set_index('Date', inplace=True)
#check datatype of index
left_likes_filtered.index                                 #converting object type of month to timestamp of dataframe of datatime64[ns]


# In[358]:


left_likes_ts = left_likes_filtered['likes_count']
left_likes_ts.head(10)


# In[359]:


from statsmodels.tsa.stattools import adfuller
def check_stationarity(ts):
    dftest = adfuller(ts)
    adf = dftest[0]
    pvalue = dftest[1]
    critical_value = dftest[4]['5%']
    if (pvalue < 0.05) and (adf < critical_value):      #here we are defining a function that uses Dicky fuller method and prints the results based on pvalue and test static value
        print('The series is stationary')
    else:
        print('The series is NOT stationary')


# In[360]:


left_likes_ts=left_likes_ts.sort_index(ascending=True)


# In[361]:


check_stationarity(left_likes_ts)


# In[362]:


plt.plot(left_likes_ts)


# In[363]:


result = seasonal_decompose(left_likes_ts,model='additive')      #applied seasonal decomposition to get trend and seasons.
fig = result.plot()


# In[364]:


plot_acf(left_likes_ts, lags=20);
plot_pacf(left_likes_ts, lags=20);  #getting order of Auto regression (p) and getting number of forecast errors(q) to specify for the model.
                         #p is point where acf increased significantly, it can be any points but in our graph we get increasing trend at 0.


# In[365]:


from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
 
# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]
 
# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error
 
# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)
 # grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores
 
# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1,2]
    q_params = [0, 1,2]
    t_params = ['n','c','t','ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
    return models


# In[366]:


left_likes_ts.info()


# In[367]:


n_test = 36
    # model configs
cfg_list = sarima_configs()
    # grid search
scores = grid_search(left_likes_ts, cfg_list, n_test)
print('done')
    # list top 3 configs
for cfg, error in scores[:3]:
    print(cfg, error)


# In[368]:


sarima = SARIMAX(left_likes_ts, order=(11,1,6), seasonal_order=(3,0,4,12))
sarima_predictions = sarima.fit().predict()                            #Links to get SARIMAX parameters :- https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
                                                                       #https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
                                                 


# In[369]:


plt.figure(figsize=(16,4))
plt.plot(left_likes_ts, label="Actual")
plt.plot(sarima_predictions, label="Predicted")
plt.title('No of left likes', fontsize=20)
#plt.ylabel('Sales', fontsize=16)
plt.legend()


# In[370]:


print('RMSE OF SARIMA of LEFT Likes count: %.4f'%np.sqrt(sum((left_likes_ts-sarima_predictions)**2)/(len(left_likes_ts))))


# In[371]:


#geting righttweet likes count by SARIMA


# In[372]:


right_likes_dataset['Date'] = pd.to_datetime(right_likes_dataset['Date'], format='%Y-%m-%d')


# In[373]:


right_likes_filtered=right_likes_dataset.loc[(right_likes_dataset['Date'] >= '2018-05-01') & (right_likes_dataset['Date'] <= '2018-12-31')]


# In[374]:


from datetime import datetime
con=right_likes_filtered['Date']
right_likes_filtered['Date']=pd.to_datetime(right_likes_filtered['Date'])
right_likes_filtered.set_index('Date', inplace=True)
#check datatype of index
right_likes_filtered.index    


# In[375]:


right_likes_ts = right_likes_filtered['likes_count']
right_likes_ts.head(10)


# In[376]:


right_likes_ts=right_likes_ts.sort_index(ascending=True)


# In[377]:


check_stationarity(right_likes_ts)


# In[378]:


plt.plot(right_likes_ts)


# In[379]:


right_likes_ts.info()


# In[380]:


result = seasonal_decompose(right_likes_ts,model='additive',period=20)     #applied seasonal decomposition to get trend and seasons.
fig = result.plot()


# In[381]:


n_test = 36
    # model configs
cfg_list = sarima_configs()
    # grid search
scores = grid_search(right_likes_ts, cfg_list, n_test)
print('done')
    # list top 3 configs
for cfg, error in scores[:3]:
    print(cfg, error)


# In[382]:


plot_acf(right_likes_ts, lags=20);
plot_pacf(right_likes_ts, lags=20);  #getting order of Auto regression (p) and getting number of forecast errors(q) to specify for the model.
                         #p is point where acf increased significantly, it can be any points but in our graph we get increasing trend at 0.


# In[383]:


sarima = SARIMAX(right_likes_ts, order=(9,1,11), seasonal_order=(1,1,3,12))
sarima_predictions = sarima.fit().predict()                            #Links to get SARIMAX parameters :- https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
                                                                       #https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
                                    


# In[384]:


plt.figure(figsize=(16,4))
plt.plot(right_likes_ts, label="Actual")
plt.plot(sarima_predictions, label="Predicted")
plt.title('No of right likes', fontsize=20)
#plt.ylabel('Sales', fontsize=16)
plt.legend()


# In[385]:


print('RMSE OF SARIMA of Right Likes count: %.4f'%np.sqrt(sum((right_likes_ts-sarima_predictions)**2)/(len(right_likes_ts))))


# In[386]:


# getting predictions for centre likes dataset


# In[387]:


centre_likes_dataset['Date'] = pd.to_datetime(centre_likes_dataset['Date'], format='%Y-%m-%d')


# In[388]:


centre_likes_filtered=centre_likes_dataset.loc[(centre_likes_dataset['Date'] >= '2018-05-01') & (centre_likes_dataset['Date'] <= '2018-12-31')]


# In[389]:


from datetime import datetime
con=centre_likes_filtered['Date']
centre_likes_filtered['Date']=pd.to_datetime(centre_likes_filtered['Date'])
centre_likes_filtered.set_index('Date', inplace=True)
#check datatype of index
centre_likes_filtered.index  


# In[390]:


centre_likes_ts = centre_likes_filtered['likes_count']
centre_likes_ts.head(10)


# In[391]:


centre_likes_ts.info()


# In[392]:


centre_likes_ts=centre_likes_ts.sort_index(ascending=True)


# In[393]:


check_stationarity(centre_likes_ts)


# In[394]:


plt.plot(centre_likes_ts)


# In[395]:


result = seasonal_decompose(centre_likes_ts,model='additive',period=10)     #applied seasonal decomposition to get trend and seasons.
fig = result.plot()


# In[396]:


plot_acf(centre_likes_ts, lags=20);
plot_pacf(centre_likes_ts, lags=20);  #getting order of Auto regression (p) and getting number of forecast errors(q) to specify for the model.
                         #p is point where acf increased significantly, it can be any points but in our graph we get increasing trend at 0.


# In[397]:


n_test = 32
    # model configs
cfg_list = sarima_configs()
    # grid search
scores = grid_search(centre_likes_ts, cfg_list, n_test)
print('done')
    # list top 3 configs
for cfg, error in scores[:3]:
    print(cfg, error)


# In[398]:


sarima = SARIMAX(centre_likes_ts, order=(8,1,11), seasonal_order=(4,0,0,12))
sarima_predictions = sarima.fit().predict()                            #Links to get SARIMAX parameters :- https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
                                                                       #https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
                                    


# In[399]:


plt.figure(figsize=(16,4))
plt.plot(centre_likes_ts, label="Actual")
plt.plot(sarima_predictions, label="Predicted")
plt.title('No of centre likes', fontsize=20)
#plt.ylabel('Sales', fontsize=16)
plt.legend()


# In[400]:


print('RMSE OF SARIMA of centre Likes count: %.4f'%np.sqrt(sum((centre_likes_ts-sarima_predictions)**2)/(len(centre_likes_ts))))


# In[401]:


#LSTM model for left likes count for lookback=1


# In[402]:


dataset=left_likes_ts


# In[403]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1,1))
train_size = int(len(dataset) * 0.7)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[404]:


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):   #The function takes two arguments: the dataset, which is a NumPy array that we want to convert into a dataset, 
  #and the look_back, which is the number of previous time steps to use as input variables to predict the next time period — in this case defaulted to 1.
#This default will create a dataset where X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[405]:


from tensorflow import keras
from keras_tuner.tuners import BayesianOptimization
import os
n_input = 1
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units',min_value=32,
                                    max_value=512,
                                    step=32), 
               activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(units=hp.Int('units',min_value=32,
                                    max_value=512,
                                    step=32), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', metrics=['mse'], optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate',
                  values=[1e-2, 1e-3, 1e-4])))

    return model

bayesian_opt_tuner = BayesianOptimization(
    build_model,
    objective='mse',
    max_trials=3, 
    executions_per_trial=1,
    directory=os.path.normpath('C:/keras_tuning'),
    project_name='kerastuner_bayesian_poc',
    overwrite=True)

bayesian_opt_tuner.search(train, train,epochs=60,
     #validation_data=(X_test, y_test)
     validation_split=0.2,verbose=1)


bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)
model = bayes_opt_model_best_model[0]
bayesian_opt_tuner.results_summary()
bayesian_opt_tuner.search_space_summary()


# In[406]:


look_back = 1
trainX, trainY = create_dataset(train, look_back) #converting as above mentioned function for training and testing.
testX, testY = create_dataset(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))     


# In[407]:


from keras import backend as K


# In[408]:


model = Sequential()
model.add(LSTM(units=4, input_shape=(1, look_back)))#4 are the hidden LSTM blocks and 1 input layer as visible.
#model.add(Dense(units=512))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#K.set_value(model.optimizer.learning_rate, 0.001)
model.fit(trainX, trainY, epochs=75, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;


# In[409]:


model.optimizer.get_config()


# In[410]:


model.summary()


# In[411]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[412]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[413]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score RMSE OF LEFT Likes : %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score RMSE OF LEFT Likes: %.2f RMSE' % (testScore))


# In[414]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)         
trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[415]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
plt.show()


# In[416]:


#LSTM Model for right likes count, lookback=1


# In[466]:


dataset=right_likes_ts


# In[467]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1,1))
train_size = int(len(dataset) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[419]:


from tensorflow import keras
from keras_tuner.tuners import BayesianOptimization
import os
n_input = 1
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units',min_value=32,
                                    max_value=512,
                                    step=32), 
               activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(units=hp.Int('units',min_value=32,
                                    max_value=512,
                                    step=32), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', metrics=['mse'], optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate',
                  values=[1e-2, 1e-3, 1e-4])))

    return model

bayesian_opt_tuner = BayesianOptimization(
    build_model,
    objective='mse',
    max_trials=3, 
    executions_per_trial=1,
    directory=os.path.normpath('C:/keras_tuning'),
    project_name='kerastuner_bayesian_poc',
    overwrite=True)

bayesian_opt_tuner.search(train, train,epochs=100,
     #validation_data=(X_test, y_test)
     validation_split=0.2,verbose=1)


bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)
model = bayes_opt_model_best_model[0]
bayesian_opt_tuner.results_summary()
bayesian_opt_tuner.search_space_summary()


# In[468]:


look_back = 1
trainX, trainY = create_dataset(train, look_back) #converting as above mentioned function for training and testing.
testX, testY = create_dataset(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))     


# In[469]:


model = Sequential()
model.add(LSTM(units=32, input_shape=(1, look_back)))#4 are the hidden LSTM blocks and 1 input layer as visible.
#model.add(Dense(units=5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
K.set_value(model.optimizer.learning_rate, 0.01)
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;


# In[422]:


model.optimizer.get_config()


# In[423]:


model.summary()


# In[470]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[471]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[472]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score RMSE OF right Likes : %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score RMSE OF right Likes: %.2f RMSE' % (testScore))


# In[473]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)         
trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[474]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
plt.show()


# In[429]:


#LSTM model for centre likes, feedback=1


# In[493]:


dataset=centre_likes_ts


# In[494]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1,1))
train_size = int(len(dataset) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[432]:


from tensorflow import keras
from keras_tuner.tuners import BayesianOptimization
import os
n_input = 1
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units',min_value=32,
                                    max_value=512,
                                    step=32), 
               activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(units=hp.Int('units',min_value=32,
                                    max_value=512,
                                    step=32), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', metrics=['mse'], optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate',
                  values=[1e-2, 1e-3, 1e-4])))

    return model

bayesian_opt_tuner = BayesianOptimization(
    build_model,
    objective='mse',
    max_trials=3, 
    executions_per_trial=1,
    directory=os.path.normpath('C:/keras_tuning'),
    project_name='kerastuner_bayesian_poc',
    overwrite=True)

bayesian_opt_tuner.search(train, train,epochs=50,
     #validation_data=(X_test, y_test)
     validation_split=0.2,verbose=1)


bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)
model = bayes_opt_model_best_model[0]
bayesian_opt_tuner.results_summary()
bayesian_opt_tuner.search_space_summary()


# In[495]:


look_back = 1
trainX, trainY = create_dataset(train, look_back) #converting as above mentioned function for training and testing.
testX, testY = create_dataset(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[496]:


model = Sequential()
model.add(LSTM(units=64, input_shape=(1, look_back)))#4 are the hidden LSTM blocks and 1 input layer as visible.
#model.add(Dense(units=2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#K.set_value(model.optimizer.learning_rate, 0.01)
model.fit(trainX, trainY, epochs=65, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;


# In[503]:


model.optimizer.get_config()


# In[502]:


model.summary()


# In[497]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[498]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[499]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score RMSE OF centre Likes : %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score RMSE OF centre Likes: %.2f RMSE' % (testScore))


# In[500]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)         
trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[501]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
plt.show()


# In[442]:


#LSTM forecasting for left likes for feedback=3


# In[504]:


dataset1=left_likes_ts


# In[505]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(dataset1.values.reshape(-1,1))
train_size = int(len(dataset1) * 0.7)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset1) - train_size
train, test = dataset1[0:train_size,:], dataset1[train_size:len(dataset1),:]
print(len(train), len(test))


# In[506]:


for i in range(3,4):
    trainX, trainY = create_dataset(train, i) #converting as above mentioned function for training and testing.
    testX, testY = create_dataset(test, i)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(units=4, input_shape=(1, i))) #4 are the hidden LSTM blocks and 1 input layer as visible.
    #model.add(Dense(units=4))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=75, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score RMSE OF left tweets: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score RMSE OF left tweets: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset1)         
    trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
    trainPredictPlot[i:len(trainPredict)+i, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(i*2)+1:len(dataset1)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset1))
    plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
    plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
    plt.show()
    model.summary()


# In[446]:


#LSTM forecasting for right likes , for feedeback=3;


# In[507]:


dataset2=right_likes_ts


# In[508]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset2 = scaler.fit_transform(dataset2.values.reshape(-1,1))
train_size = int(len(dataset2) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset2) - train_size
train, test = dataset2[0:train_size,:], dataset2[train_size:len(dataset2),:]
print(len(train), len(test))


# In[509]:


for i in range(3,4):
    trainX, trainY = create_dataset(train, i) #converting as above mentioned function for training and testing.
    testX, testY = create_dataset(test, i)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, i))) #4 are the hidden LSTM blocks and 1 input layer as visible.
    #model.add(Dense(units=4))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=75, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score RMSE OF right likes: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score RMSE OF right likes: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset1)         
    trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
    trainPredictPlot[i:len(trainPredict)+i, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset2)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(i*2)+1:len(dataset2)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset2))
    plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
    plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
    plt.show()
    model.summary()


# In[450]:


#LSTM for centre likes for feedback=2


# In[513]:


dataset3=centre_likes_ts


# In[514]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset3 = scaler.fit_transform(dataset3.values.reshape(-1,1))
train_size = int(len(dataset3) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset3) - train_size
train, test = dataset3[0:train_size,:], dataset3[train_size:len(dataset3),:]
print(len(train), len(test))


# In[515]:


for i in range(3,4):
    trainX, trainY = create_dataset(train, i) #converting as above mentioned function for training and testing.
    testX, testY = create_dataset(test, i)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(units=32, input_shape=(1, i))) #4 are the hidden LSTM blocks and 1 input layer as visible.
    #model.add(Dense(units=2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=65, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score RMSE OF centre likes: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score RMSE OF centre likes: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset1)         
    trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
    trainPredictPlot[i:len(trainPredict)+i, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset3)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(i*2)+1:len(dataset3)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset3))
    plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
    plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
    plt.show()
    model.summary()


# In[454]:


#Multistep forecasting for left likes


# In[516]:


def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
 
# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test
 
# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    #model.add(Dense(n_neurons))
    model.add(Dense(1))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2, shuffle=False)
        model.reset_states()
    return model
# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]
 
# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts
 
# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted
 
# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted
# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))
 
# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    pyplot.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
    # show the plot
    pyplot.show()


# In[519]:


# load dataset
series = left_likes_ts
# configure
n_lag = 3
n_seq = 6
n_test = 55
n_epochs = 65
n_batch = 1
n_neurons = 16
# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(series, forecasts, n_test+2)


# In[457]:


model.summary()


# In[458]:


#multistep forecasting for right likes


# In[521]:


# load dataset
series = right_likes_ts
# configure
n_lag = 3
n_seq = 6
n_test = 37
n_epochs = 15
n_batch = 1
n_neurons = 16
# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(series, forecasts, n_test+2)


# In[460]:


model.summary()


# In[461]:


#multistep forecasting for centre likes


# In[524]:


# load dataset
series = centre_likes_ts
# configure
n_lag = 3
n_seq = 6
n_test = 37
n_epochs = 25
n_batch = 1
n_neurons = 32
# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(series, forecasts, n_test+2)


# In[525]:


model.summary()

