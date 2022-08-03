#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# In[2]:


df=pd.read_json("tweet_ICWSM.json",lines=True)


# In[3]:


df.head()


# In[4]:


print(df.size)
df.dtypes


# In[5]:


df1=df['entities']


# In[6]:


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)


# In[7]:


df1.head()


# In[8]:


df1.dtypes


# In[9]:


expanded_urls = df['entities'].apply(lambda x: (x['urls'][0]['expanded_url'] if len(x['urls'])> 0 else pd.NA))
expanded_urls = expanded_urls[expanded_urls.notna()]


# In[10]:


expanded_urls.tolist()


# In[11]:


df['expanded_urls']=expanded_urls


# In[12]:


df.head()


# In[13]:


df_required=df[['created_at','expanded_urls']]


# In[14]:


df_required.head()


# In[15]:


df_required.head()


# In[16]:


df_required.size


# In[17]:


data2=pd.DataFrame(pd.read_excel("MediaBias.xlsx"))


# In[18]:


data3 = data2[['Media source', 'URL', 'Political Leaning']].copy()
data3.dtypes


# In[19]:


#SPLITTING THE ENTIRE DATASET OF MEDIABIAS INTO LEFT,CENTRE AND RIGHT


# In[20]:


left_media=[]
right_media=[]
centre_media=[]


# In[21]:


data3.rename(columns = {'Media source':'Media_source', 'Political Leaning':'Political_Leaning'}, inplace = True)
data3.Media_source = data3.Media_source.astype('string')
data3.Political_Leaning=data3.Political_Leaning.astype('string')
data3.URL=data3.URL.astype('string')


# In[22]:


data3.dropna(inplace=True)


# In[23]:


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


# In[24]:


print(len(left_media))
print(len(right_media))
print(len(centre_media))


# In[25]:


hashmap_media={}


# In[26]:


for ind in data3.index:
    hashmap_media[data3['Media_source'][ind]]=data3['URL'][ind]


# In[27]:


left_timestamps=[]
right_timestamps=[]
centre_timestamps=[]


# In[28]:


import re


# In[29]:


df_required.isnull().values.any()


# In[30]:


left_suburls=[]
right_suburls=[]
centre_suburls=[]


# In[31]:


for i in range(0,len(left_media)):
    if hashmap_media.get(left_media[i]) not in left_suburls:
        left_suburls.append(hashmap_media.get(left_media[i]))


# In[32]:


print(len(left_suburls))
print((left_suburls))


# In[33]:


for i in range(0,len(right_media)):
    if hashmap_media.get(right_media[i]) not in right_suburls:
        right_suburls.append(hashmap_media.get(right_media[i]))


# In[34]:


print(len(right_suburls))
print(right_suburls)


# In[35]:


for i in range(0,len(centre_media)):
    if hashmap_media.get(centre_media[i]) not in centre_suburls:
        centre_suburls.append(hashmap_media.get(centre_media[i]))


# In[36]:


print(len(centre_suburls))


# In[37]:


left_timestamps=[]
right_timestamps=[]
centre_timestamps=[]


# In[38]:


x=df_required[["created_at","expanded_urls"]].to_numpy()


# In[39]:


print(x)
print(len(x))
print(type(x[1]))
left=np.array(left_suburls)


# In[40]:


for y in x:
    for sub  in left_suburls:
        if y[1] is not np.nan:
            if sub in str(y[1]):
                left_timestamps.append(y[0].date())
        


# In[41]:


for y in x:
    for sub  in right_suburls:
        if y[1] is not np.nan:
            if sub in str(y[1]):
                right_timestamps.append(y[0].date())
        


# In[42]:


for y in x:
    for sub  in centre_suburls:
        if y[1] is not np.nan:
            if sub in str(y[1]):
                centre_timestamps.append(y[0].date())


# In[43]:


print(len(left_timestamps))
print(len(right_timestamps))
print(len(centre_timestamps))


# In[44]:


#Creating hashmap of frequency of tweets


# In[45]:


df_left_timestamps=pd.DataFrame(left_timestamps)


# In[46]:


df_left_timestamps['timestamps']=pd.DataFrame(left_timestamps)


# In[47]:


df_left_timestamps.head()


# In[48]:


df_right_timestamps=pd.DataFrame(right_timestamps)


# In[49]:


df_right_timestamps['timestamps']=pd.DataFrame(right_timestamps)


# In[50]:


df_right_timestamps.head()


# In[51]:


df_centre_timestamps=pd.DataFrame(centre_timestamps)


# In[52]:


df_centre_timestamps['timestamps']=pd.DataFrame(centre_timestamps)


# In[53]:


df_centre_timestamps.head()


# In[54]:


left_timestamp_frequency={}
right_timestamp_frequency={}
centre_timestamp_frequency={}


# In[55]:


for i in  df_left_timestamps.index :
    if df_left_timestamps['timestamps'][i] not in left_timestamp_frequency:
        left_timestamp_frequency[df_left_timestamps['timestamps'][i]]=1
    else:
        left_timestamp_frequency[df_left_timestamps['timestamps'][i]]+=1


# In[56]:


print(len(left_timestamp_frequency))


# In[57]:


for i in  df_right_timestamps.index :
    if df_right_timestamps['timestamps'][i] not in right_timestamp_frequency:
        right_timestamp_frequency[df_right_timestamps['timestamps'][i]]=1
    else:
        right_timestamp_frequency[df_right_timestamps['timestamps'][i]]+=1


# In[58]:


print(len(right_timestamp_frequency))


# In[59]:


for i in  df_centre_timestamps.index :
    if df_centre_timestamps['timestamps'][i] not in centre_timestamp_frequency:
        centre_timestamp_frequency[df_centre_timestamps['timestamps'][i]]=1
    else:
        centre_timestamp_frequency[df_centre_timestamps['timestamps'][i]]+=1


# In[60]:


print(len(centre_timestamp_frequency))


# In[61]:


#creating dataframes for models


# In[62]:


l1=[]
l2=[]


# In[63]:


for key in left_timestamp_frequency:
    l1.append(key)
    l2.append(left_timestamp_frequency[key])


# In[64]:


l = {'Date':l1,'frequency':l2}
left_dataset=pd.DataFrame(l)


# In[65]:


left_dataset.head()


# In[66]:


left_dataset['Date'] = pd.to_datetime(left_dataset['Date'], format='%Y-%m-%d')
  


# In[67]:


left_filtered=left_dataset.loc[(left_dataset['Date'] >= '2018-05-01') & (left_dataset['Date'] <= '2018-12-31')]


# In[68]:


left_filtered=left_filtered.drop_duplicates()


# In[69]:


print(left_filtered.size)


# In[70]:


left_filtered.head()


# In[71]:


r1=[]
r2=[]


# In[72]:


for key in right_timestamp_frequency:
    r1.append(key)
    r2.append(right_timestamp_frequency[key])


# In[73]:


r={'Date':r1,'frequency':r2}
right_dataset=pd.DataFrame(r)


# In[74]:


right_dataset.head()


# In[75]:


print(right_dataset.size)


# In[76]:


right_dataset['Date'] = pd.to_datetime(right_dataset['Date'], format='%Y-%m-%d')


# In[77]:


right_filtered=right_dataset.loc[(right_dataset['Date'] >= '2018-05-01') & (right_dataset['Date'] <='2018-12-31')]


# In[78]:


right_filtered=right_filtered.drop_duplicates()


# In[79]:


print(right_filtered.size)


# In[80]:


c1=[]
c2=[]


# In[81]:


for key in centre_timestamp_frequency:
    c1.append(key)
    c2.append(centre_timestamp_frequency[key])


# In[82]:


c={'Date':c1,'frequency':c2}
centre_dataset=pd.DataFrame(c)


# In[83]:


centre_dataset.head()


# In[84]:


print(centre_dataset.size)


# In[85]:


centre_dataset['Date'] = pd.to_datetime(centre_dataset['Date'], format='%Y-%m-%d')


# In[86]:


centre_filtered=centre_dataset.loc[(centre_dataset['Date'] >= '2018-05-01') & (centre_dataset['Date'] <= '2018-12-31')]


# In[87]:


centre_filtered=centre_filtered.drop_duplicates()


# In[88]:


print(centre_filtered.size)


# In[89]:


#Left tweets prediction


# In[90]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[91]:


from datetime import datetime


# In[92]:


left_filtered['Date']=pd.to_datetime(left_filtered['Date'])
left_filtered.set_index('Date', inplace=True)
#check datatype of index
left_filtered.index                              


# In[93]:


left_ts = left_filtered['frequency']
left_ts.head(10)


# In[94]:


left_ts=left_ts.sort_index(ascending=True)


# In[95]:


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


# In[96]:


check_stationarity(left_ts)


# In[97]:


plt.plot(left_ts)


# In[98]:


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


# In[99]:


result = seasonal_decompose(left_ts,model='additive',period=10)      #applied seasonal decomposition to get trend and seasons.
fig = result.plot()


# In[100]:


plot_acf(left_ts, lags=20);
plot_pacf(left_ts, lags=20);  #getting order of Auto regression (p) and getting number of forecast errors(q) to specify for the model.
                         #p is point where acf increased significantly, it can be any points but in our graph we get increasing trend at 0.


# In[101]:


sarima = SARIMAX(left_ts, order=(9,0,10), seasonal_order=(2,1,1,12))
sarima_predictions = sarima.fit().predict()                            #Links to get SARIMAX parameters :- https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
                                                                       #https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
                                                                       #https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/


# In[102]:


plt.figure(figsize=(16,4))
plt.plot(left_ts, label="Actual")
plt.plot(sarima_predictions, label="Predicted")
plt.title('frequency of tweets', fontsize=20)
#plt.ylabel('Sales', fontsize=16)
plt.legend()


# In[103]:


print('RMSE OF SARIMA OF LEFT TWEETS: %.4f'%np.sqrt(sum((left_ts-sarima_predictions)**2)/(len(left_ts))))


# In[104]:


#right tweets prediction


# In[105]:


right_filtered['Date']=pd.to_datetime(right_filtered['Date'])
right_filtered.set_index('Date', inplace=True)
#check datatype of index
right_filtered.index      


# In[106]:


right_ts = right_filtered['frequency']
right_ts.head(10)


# In[107]:


right_ts=right_ts.sort_index(ascending=True)


# In[108]:


check_stationarity(right_ts)


# In[109]:


plt.plot(right_ts)


# In[110]:


result = seasonal_decompose(right_ts,model='additive',period=10)      #applied seasonal decomposition to get trend and seasons.
fig = result.plot()


# In[111]:


plot_acf(right_ts, lags=20);
plot_pacf(right_ts, lags=20);  


# In[112]:


sarima = SARIMAX(right_ts, order=(7,0,10), seasonal_order=(2,1,1,12))
sarima_predictions = sarima.fit().predict()                            #Links to get SARIMAX parameters :- https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
                                            


# In[113]:


plt.figure(figsize=(16,4))
plt.plot(right_ts, label="Actual")
plt.plot(sarima_predictions, label="Predicted")
plt.title('frequency of tweets', fontsize=20)
#plt.ylabel('Sales', fontsize=16)
plt.legend()


# In[114]:


print('RMSE OF SARIMA OF RIGHT TWEETS: %.4f'%np.sqrt(sum((right_ts-sarima_predictions)**2)/(len(right_ts))))


# In[115]:


#centre data predictions


# In[116]:


centre_filtered['Date']=pd.to_datetime(centre_filtered['Date'])
centre_filtered.set_index('Date', inplace=True)
#check datatype of index
centre_filtered.index    


# In[117]:


centre_ts = centre_filtered['frequency']
centre_ts.head(10)


# In[118]:


centre_ts=centre_ts.sort_index(ascending=True)


# In[119]:


check_stationarity(centre_ts)


# In[120]:


plt.plot(centre_ts)


# In[121]:


result = seasonal_decompose(right_ts,model='additive',period=10)      #applied seasonal decomposition to get trend and seasons.
fig = result.plot()


# In[122]:


plot_acf(centre_ts, lags=20);
plot_pacf(centre_ts, lags=20);  


# In[123]:


sarima = SARIMAX(centre_ts, order=(9,0,10), seasonal_order=(2,1,1,12))
sarima_predictions = sarima.fit().predict()    


# In[124]:


plt.figure(figsize=(16,4))
plt.plot(centre_ts, label="Actual")
plt.plot(sarima_predictions, label="Predicted")
plt.title('frequency of tweets', fontsize=20)
#plt.ylabel('Sales', fontsize=16)
plt.legend()


# In[125]:


print('RMSE OF SARIMA OF CENTRE TWEETS: %.4f'%np.sqrt(sum((centre_ts-sarima_predictions)**2)/(len(centre_ts))))


# In[126]:


left_dataset.describe


# In[127]:


right_dataset.describe


# In[128]:


centre_dataset.describe


# In[129]:


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


# In[130]:


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


# In[131]:


#LSTM MODEL FOR LEFT TWEETS PREDICTION


# In[132]:


#left_filtered1=left_dataset.loc[(left_dataset['Date'] >= '2018-05-01') & (left_dataset['Date'] <= '2018-12-31')]


# In[133]:


#left_filtered1=left_filtered1.drop_duplicates()


# In[134]:


#left_filtered1.to_csv('data1.csv')


# In[135]:


#left_lstm=pd.read_csv('data1.csv')


# In[136]:


#left_lstm1=left_lstm[['Date','frequency']]


# In[137]:


#left_lstm1.describe


# In[138]:


#dataset = left_lstm1['frequency'].values.reshape(-1,1)
#dataset = dataset.astype('float32')


# In[139]:


dataset=left_ts


# In[140]:


print(left_ts)


# In[141]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1,1))
train_size = int(len(dataset) * 0.67)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[142]:


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):   #The function takes two arguments: the dataset, which is a NumPy array that we want to convert into a dataset, 
  #and the look_back, which is the number of previous time steps to use as input variables to predict the next time period — in this case defaulted to 1.
#This default will create a dataset where X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[143]:


#hyper parameter tunning for Left tweets LSTM model


# In[144]:


import tensorflow as f
from tensorflow import keras
import keras_tuner as kt


# In[145]:


import numpy as np
from keras.layers import LSTM,Input
from keras.models import Sequential


# In[146]:


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
bayesian_opt_tuner.results_summary()


# In[147]:


look_back = 1
trainX, trainY = create_dataset(train, look_back) #converting as above mentioned function for training and testing.
testX, testY = create_dataset(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))     


# In[148]:


from keras import backend as K


# In[149]:


model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back))) #4 are the hidden LSTM blocks and 1 input layer as visible.
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#K.set_value(model.optimizer.learning_rate, 0.01)
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;


# In[150]:


model.summary()


# In[151]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[152]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[153]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score RMSE OF LEFT TWEETS: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score RMSE OF LEFT TWEETS: %.2f RMSE' % (testScore))


# In[154]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)         
trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[155]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
plt.show()


# In[156]:


#centre tweets prediction by LSTM MODEL


# In[157]:


#centre_filtered1=centre_dataset.loc[(centre_dataset['Date'] >= '2018-05-01') & (centre_dataset['Date'] <= '2018-12-31')]


# In[158]:


#centre_filtered1=centre_filtered1.drop_duplicates()


# In[159]:


#centre_filtered1.describe


# In[160]:


#centre_filtered1.to_csv('data2.csv')


# In[161]:


#dataframe = pandas.read_csv('data2.csv', usecols=[2], engine='python')
#dataset = dataframe.values
#dataset = dataset.astype('float32')


# In[162]:


dataset=centre_ts


# In[163]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1,1))


# In[164]:


train_size = int(len(dataset) * 0.67)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[165]:


#centre hyper parameters optimization


# In[166]:


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


# In[167]:


look_back = 1
trainX, trainY = create_dataset(train, look_back) #converting as above mentioned function for training and testing.
testX, testY = create_dataset(test, look_back)


# In[168]:


trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))   


# In[169]:


model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back))) #4 are the hidden LSTM blocks and 1 input layer as visible.
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='RMSProp')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;


# In[170]:


model.summary()


# In[171]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[172]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[173]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score RMSE OF centre tweets: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score RMSE OF centre tweets: %.2f RMSE' % (testScore))


# In[174]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)         
trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[175]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
plt.show()


# In[176]:


#right tweet prediction using LSTM.


# In[177]:


#right_filtered1=right_dataset.loc[(right_dataset['Date'] >= '2018-05-01') & (right_dataset['Date'] <= '2018-12-31')]


# In[178]:


#right_filtered1=right_filtered1.drop_duplicates()


# In[179]:


#right_filtered1.to_csv('data3.csv')


# In[180]:


#dataframe = pandas.read_csv('data3.csv', usecols=[2], engine='python')
#dataset = dataframe.values
#dataset = dataset.astype('float32')


# In[181]:


dataset=right_ts


# In[182]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1,1))    


# In[183]:


train_size = int(len(dataset) * 0.67)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[184]:


look_back = 1
trainX, trainY = create_dataset(train, look_back) #converting as above mentioned function for training and testing.
testX, testY = create_dataset(test, look_back)


# In[185]:


trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))       


# In[186]:


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


# In[187]:


model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back))) #4 are the hidden LSTM blocks and 1 input layer as visible.
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='RMSProp')
model.fit(trainX, trainY, epochs=96, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;


# In[188]:


model.summary()


# In[189]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[190]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[191]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score RMSE OF right tweets: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score RMSE OF right tweets: %.2f RMSE' % (testScore))


# In[192]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)         
trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[193]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
plt.show()


# In[194]:


#left tweets prediction using multiple lookbacks.


# In[195]:


dataset1=left_ts


# In[196]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(dataset1.values.reshape(-1,1))
train_size = int(len(dataset1) * 0.67)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset1) - train_size
train, test = dataset1[0:train_size,:], dataset1[train_size:len(dataset1),:]
print(len(train), len(test))


# In[197]:


look_backs=[]
for i in range(2,11):
    look_backs.append(i)


# In[198]:


for i in range(2,3):
    trainX, trainY = create_dataset(train, i) #converting as above mentioned function for training and testing.
    testX, testY = create_dataset(test, i)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, i))) #4 are the hidden LSTM blocks and 1 input layer as visible.
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='RMSProp')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;
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


# In[199]:


#centre tweets prediction for multiple lookbacks


# In[200]:


dataset2=centre_ts


# In[201]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset2 = scaler.fit_transform(dataset2.values.reshape(-1,1))
train_size = int(len(dataset1) * 0.67)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset2) - train_size
train, test = dataset2[0:train_size,:], dataset2[train_size:len(dataset2),:]
print(len(train), len(test))


# In[202]:


for i in range(2,3):
    trainX, trainY = create_dataset(train, i) #converting as above mentioned function for training and testing.
    testX, testY = create_dataset(test, i)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, i))) #4 are the hidden LSTM blocks and 1 input layer as visible.
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='RMSProp')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;
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


# In[203]:


# right tweets prediction for multiple lookbacks


# In[204]:


dataset3=right_ts


# In[205]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset3 = scaler.fit_transform(dataset3.values.reshape(-1,1))
train_size = int(len(dataset3) * 0.67)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset3) - train_size
train, test = dataset3[0:train_size,:], dataset3[train_size:len(dataset2),:]
print(len(train), len(test))


# In[206]:


for i in range(2,3):
    trainX, trainY = create_dataset(train, i) #converting as above mentioned function for training and testing.
    testX, testY = create_dataset(test, i)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, i))) #4 are the hidden LSTM blocks and 1 input layer as visible.
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=96, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;
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


# In[207]:


# multistep time series of lstm


# In[208]:


#left tweets prediction in multiple time series of lstm


# In[209]:


dataset4=left_filtered
dataset4.describe


# In[210]:


left_ts.describe


# In[211]:


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


# In[212]:


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
 


# In[213]:


# load dataset
series = left_ts
# configure
n_lag = 2
n_seq = 6
n_test = 53
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


# In[214]:


model.summary()


# In[215]:


#right multistep time series of lstm


# In[216]:


# load dataset
series = right_ts
# configure
n_lag = 2
n_seq = 6
n_test = 51
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


# In[217]:


model.summary()


# In[218]:


# load dataset
series = centre_ts
# configure
n_lag = 2
n_seq = 6
n_test = 53
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


# In[219]:


model.summary()


# In[220]:


#adaption rate determination


# In[221]:


df.info()


# In[222]:


df_required.info()


# In[223]:


df_required['likes_count']=df['favorite_count']


# In[224]:


df_required['shares_count']=df['retweet_count']


# In[225]:


df_required.info()


# In[226]:


print(df_required.tail(13))


# In[227]:


x1=df_required[["created_at","expanded_urls","likes_count","shares_count"]].to_numpy()


# In[228]:


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


# In[229]:


#getting left like counts


# In[230]:


for y in x1:
    for sub  in left_suburls:
        if y[1] is not np.nan:
            if sub in str(y[1]):
                left_timestamps1.append(y[0].date())
                left_likes.append(y[2])
                ##left_dislikes.append(y[3])
                left_reposts.append(y[3])


# In[231]:


df_left_timestamps1=pd.DataFrame()


# In[232]:


df_left_timestamps1['timestamps']=pd.DataFrame(left_timestamps1)


# In[233]:


df_left_timestamps1['likes']=left_likes


# In[234]:


df_left_timestamps1.head()


# In[ ]:





# In[235]:


df_left_timestamps1.info()


# In[236]:


df_left_timestamps2=df_left_timestamps1.drop_duplicates(keep='first')


# In[237]:


df_left_timestamps2.info()


# In[238]:


left_likes_frequency=dict(zip(df_left_timestamps2.timestamps, df_left_timestamps2.likes))


# In[239]:


for i in range(4138,len(df_left_timestamps1)):
    left_likes_frequency[df_left_timestamps1.loc[i,"timestamps"]]+=df_left_timestamps1.loc[i,"likes"]


# In[240]:


t1=[]
like=[]


# In[241]:


for key in left_likes_frequency:
    t1.append(key)
    like.append(left_likes_frequency[key])


# In[242]:


l3 = {'Date':t1,'likes_count':like}
left_likes_dataset=pd.DataFrame(l3)


# In[243]:


left_likes_dataset.head()


# In[244]:


#getting left repost counts


# In[245]:


df_left_timestamps6=pd.DataFrame()


# In[246]:


df_left_timestamps6['timestamps']=pd.DataFrame(left_timestamps1)


# In[247]:


df_left_timestamps6['reposts']=left_reposts


# In[248]:


df_left_timestamps6.info()


# In[249]:


df_left_timestamps7=df_left_timestamps6.drop_duplicates(keep='first')


# In[250]:


df_left_timestamps7.info()


# In[251]:


left_reposts_frequency=dict(zip(df_left_timestamps7.timestamps, df_left_timestamps7.reposts))


# In[252]:


for i in range(8722,len(df_left_timestamps6)):
    left_reposts_frequency[df_left_timestamps6.loc[i,"timestamps"]]+=df_left_timestamps6.loc[i,"reposts"]


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
##left_df_adoption["DISLIKES_COUNT"]=left_dislikes_dataset["dislikes_count"]
left_df_adoption["REPOSTS_COUNT"]=left_reposts_dataset["reposts_count"]


# In[261]:


left_df_adoption.head()


# In[262]:


left_df_adoption.info()


# In[263]:


#getting right tweets like counts


# In[264]:


for y in x1:
    for sub  in right_suburls:
        if y[1] is not np.nan:
            if sub in str(y[1]):
                right_timestamps1.append(y[0].date())
                right_likes.append(y[2])
                ##right_dislikes.append(y[3])
                right_reposts.append(y[3])


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


for i in range(941,len(df_right_timestamps1)):
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


#getting right tweet reposts counts


# In[279]:


df_right_timestamps5=pd.DataFrame()


# In[280]:


df_right_timestamps5['timestamps']=pd.DataFrame(right_timestamps1)


# In[281]:


df_right_timestamps5['reposts']=right_reposts


# In[282]:


df_right_timestamps5.head()


# In[283]:


df_right_timestamps6=df_right_timestamps5.drop_duplicates(keep='first')


# In[284]:


df_right_timestamps6.info()


# In[285]:


right_reposts_frequency=dict(zip(df_right_timestamps6.timestamps, df_right_timestamps6.reposts))


# In[286]:


for i in range(1993,len(df_right_timestamps5)):
    right_reposts_frequency[df_right_timestamps5.loc[i,"timestamps"]]+=df_right_timestamps5.loc[i,"reposts"]


# In[287]:


r3=[]
rreposts=[]


# In[288]:


for key in right_reposts_frequency:
    r3.append(key)
    rreposts.append(right_reposts_frequency[key])


# In[289]:


l5= {'Date':r3,'reposts_count':rreposts}
right_reposts_dataset=pd.DataFrame(l5)


# In[290]:


right_reposts_dataset.head()


# In[291]:


#combining all in a single dataframe


# In[292]:


right_df_adoption=pd.DataFrame()


# In[293]:


right_df_adoption["date"]=right_likes_dataset["Date"]
right_df_adoption["LIKES_COUNT"]=right_likes_dataset["likes_count"]
##right_df_adoption["DISLIKES_COUNT"]=right_dislikes_dataset["dislikes_count"]
right_df_adoption["REPOSTS_COUNT"]=right_reposts_dataset["reposts_count"]


# In[294]:


right_df_adoption.head()


# In[295]:


#preparing adoption rate dataset for centre 


# In[296]:


for y in x1:
    for sub  in centre_suburls:
        if y[1] is not np.nan:
            if sub in str(y[1]):
                centre_timestamps1.append(y[0].date())
                centre_likes.append(y[2])
                ##centre_dislikes.append(y[3])
                centre_reposts.append(y[3])


# In[297]:


df_centre_timestamps1=pd.DataFrame()


# In[298]:


df_centre_timestamps1['timestamps']=pd.DataFrame(centre_timestamps1)
df_centre_timestamps1['likes']=centre_likes


# In[299]:


df_centre_timestamps1.head()


# In[300]:


df_centre_timestamps2=df_centre_timestamps1.drop_duplicates(keep='first')


# In[301]:


df_centre_timestamps2.info()


# In[302]:


centre_likes_frequency=dict(zip(df_centre_timestamps2.timestamps, df_centre_timestamps2.likes))


# In[303]:


for i in range(2517,len(df_centre_timestamps1)):
    centre_likes_frequency[df_centre_timestamps1.loc[i,"timestamps"]]+=df_centre_timestamps1.loc[i,"likes"]


# In[304]:


c1=[]
clike=[]


# In[305]:


for key in centre_likes_frequency:
    c1.append(key)
    clike.append(centre_likes_frequency[key])


# In[306]:


c3 = {'Date':c1,'likes_count':clike}
centre_likes_dataset=pd.DataFrame(c3)


# In[307]:


centre_likes_dataset.head()


# In[308]:


#getting centre reposts count


# In[309]:


df_centre_timestamps5=pd.DataFrame()


# In[310]:


df_centre_timestamps5['timestamps']=pd.DataFrame(centre_timestamps1)
df_centre_timestamps5['reposts']=centre_reposts


# In[311]:


df_centre_timestamps5.head()


# In[312]:


df_centre_timestamps6=df_centre_timestamps5.drop_duplicates(keep='first')


# In[313]:


df_centre_timestamps6.info()


# In[314]:


centre_reposts_frequency=dict(zip(df_centre_timestamps6.timestamps, df_centre_timestamps6.reposts))


# In[315]:


for i in range(4401,len(df_centre_timestamps5)):
    centre_reposts_frequency[df_centre_timestamps5.loc[i,"timestamps"]]+=df_centre_timestamps5.loc[i,"reposts"]


# In[316]:


c3=[]
creposts=[]


# In[317]:


for key in centre_reposts_frequency:
    c3.append(key)
    creposts.append(centre_reposts_frequency[key])


# In[318]:


c5 = {'Date':c3,'reposts_count':creposts}
centre_reposts_dataset=pd.DataFrame(c5)


# In[319]:


centre_reposts_dataset.head()


# In[320]:


#combining all centre counts


# In[321]:


centre_df_adoption=pd.DataFrame()


# In[322]:


centre_df_adoption["date"]=centre_likes_dataset["Date"]
centre_df_adoption["LIKES_COUNT"]=centre_likes_dataset["likes_count"]
##centre_df_adoption["DISLIKES_COUNT"]=centre_dislikes_dataset["dislikes_count"]
centre_df_adoption["REPOSTS_COUNT"]=centre_reposts_dataset["reposts_count"]


# In[323]:


centre_df_adoption.head()


# In[324]:


right_df_adoption.head()


# In[325]:


left_df_adoption.head()


# In[326]:


#LIKES COUNT FORECASTING


# In[327]:


left_likes_dataset.info()
#right_likes_dataset.info()
#centre_likes_dataset.info()


# In[328]:


# likes forecasting for left leaning dataset SARIMA


# In[329]:


left_likes_dataset['Date'] = pd.to_datetime(left_dataset['Date'], format='%Y-%m-%d')


# In[330]:


left_likes_filtered=left_likes_dataset.loc[(left_likes_dataset['Date'] >= '2018-05-01') & (left_likes_dataset['Date'] <= '2018-12-31')]


# In[331]:


#left_likes_filtered=left_likes_filtered.drop_duplicates()


# In[332]:


from datetime import datetime
con=left_likes_filtered['Date']
left_likes_filtered['Date']=pd.to_datetime(left_likes_filtered['Date'])
left_likes_filtered.set_index('Date', inplace=True)
#check datatype of index
left_likes_filtered.index                                 #converting object type of month to timestamp of dataframe of datatime64[ns]


# In[333]:


left_likes_ts = left_likes_filtered['likes_count']
left_likes_ts.head(10)


# In[334]:


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


# In[335]:


left_likes_ts=left_likes_ts.sort_index(ascending=True)


# In[336]:


check_stationarity(left_likes_ts)


# In[337]:


plt.plot(left_likes_ts)


# In[338]:


result = seasonal_decompose(left_likes_ts,model='additive')      #applied seasonal decomposition to get trend and seasons.
fig = result.plot()


# In[339]:


plot_acf(left_likes_ts, lags=20);
plot_pacf(left_likes_ts, lags=20);  #getting order of Auto regression (p) and getting number of forecast errors(q) to specify for the model.
                         #p is point where acf increased significantly, it can be any points but in our graph we get increasing trend at 0.


# In[340]:


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


# In[341]:


n_test = 140
    # model configs
cfg_list = sarima_configs()
    # grid search
scores = grid_search(left_likes_ts, cfg_list, n_test)
print('done')
    # list top 3 configs
for cfg, error in scores[:3]:
    print(cfg, error)


# In[342]:


sarima = SARIMAX(left_likes_ts, order=(11,1,3), seasonal_order=(3,1,3,12))
sarima_predictions = sarima.fit().predict()                            #Links to get SARIMAX parameters :- https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
                                                                       #https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
                                                 


# In[343]:


plt.figure(figsize=(16,4))
plt.plot(left_likes_ts, label="Actual")
plt.plot(sarima_predictions, label="Predicted")
plt.title('No of left likes', fontsize=20)
#plt.ylabel('Sales', fontsize=16)
plt.legend()


# In[344]:


print('RMSE OF SARIMA of LEFT Likes count: %.4f'%np.sqrt(sum((left_likes_ts-sarima_predictions)**2)/(len(left_likes_ts))))


# In[345]:


#geting righttweet likes count by SARIMA


# In[346]:


right_likes_dataset['Date'] = pd.to_datetime(right_likes_dataset['Date'], format='%Y-%m-%d')


# In[347]:


right_likes_filtered=right_likes_dataset.loc[(right_likes_dataset['Date'] >= '2018-05-01') & (right_likes_dataset['Date'] <= '2018-12-31')]


# In[348]:


from datetime import datetime
con=right_likes_filtered['Date']
right_likes_filtered['Date']=pd.to_datetime(right_likes_filtered['Date'])
right_likes_filtered.set_index('Date', inplace=True)
#check datatype of index
right_likes_filtered.index                                 #converting object type of month to timestamp of dataframe of datatime64[ns]


# In[349]:


right_likes_ts = right_likes_filtered['likes_count']
right_likes_ts.head(10)


# In[350]:


right_likes_ts=right_likes_ts.sort_index(ascending=True)


# In[351]:


check_stationarity(right_likes_ts)


# In[352]:


plt.plot(right_likes_ts)


# In[353]:


right_likes_ts.info()


# In[354]:


result = seasonal_decompose(right_likes_ts,model='additive',period=20)     #applied seasonal decomposition to get trend and seasons.
fig = result.plot()


# In[355]:


n_test = 12
    # model configs
cfg_list = sarima_configs()
    # grid search
scores = grid_search(right_likes_ts, cfg_list, n_test)
print('done')
    # list top 3 configs
for cfg, error in scores[:3]:
    print(cfg, error)


# In[356]:


plot_acf(right_likes_ts, lags=20);
plot_pacf(right_likes_ts, lags=20);  #getting order of Auto regression (p) and getting number of forecast errors(q) to specify for the model.
                         #p is point where acf increased significantly, it can be any points but in our graph we get increasing trend at 0.


# In[357]:


sarima = SARIMAX(right_likes_ts, order=(11,1,6), seasonal_order=(2,1,1,12))
sarima_predictions = sarima.fit().predict()                            #Links to get SARIMAX parameters :- https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
                                                                       #https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
                                    


# In[358]:


plt.figure(figsize=(16,4))
plt.plot(right_likes_ts, label="Actual")
plt.plot(sarima_predictions, label="Predicted")
plt.title('No of right likes', fontsize=20)
#plt.ylabel('Sales', fontsize=16)
plt.legend()


# In[359]:


print('RMSE OF SARIMA of Right Likes count: %.4f'%np.sqrt(sum((right_likes_ts-sarima_predictions)**2)/(len(right_likes_ts))))


# In[360]:


# getting predictions for centre likes dataset


# In[361]:


centre_likes_dataset['Date'] = pd.to_datetime(centre_likes_dataset['Date'], format='%Y-%m-%d')


# In[362]:


centre_likes_filtered=centre_likes_dataset.loc[(centre_likes_dataset['Date'] >= '2018-05-01') & (centre_likes_dataset['Date'] <= '2018-12-31')]


# In[363]:


from datetime import datetime
con=centre_likes_filtered['Date']
centre_likes_filtered['Date']=pd.to_datetime(centre_likes_filtered['Date'])
centre_likes_filtered.set_index('Date', inplace=True)
#check datatype of index
centre_likes_filtered.index  


# In[364]:


centre_likes_ts = centre_likes_filtered['likes_count']
centre_likes_ts.head(10)


# In[365]:


centre_likes_ts.info()


# In[366]:


centre_likes_ts=centre_likes_ts.sort_index(ascending=True)


# In[367]:


check_stationarity(centre_likes_ts)


# In[368]:


plt.plot(centre_likes_ts)


# In[369]:


result = seasonal_decompose(centre_likes_ts,model='additive',period=10)     #applied seasonal decomposition to get trend and seasons.
fig = result.plot()


# In[370]:


plot_acf(centre_likes_ts, lags=20);
plot_pacf(centre_likes_ts, lags=20);  #getting order of Auto regression (p) and getting number of forecast errors(q) to specify for the model.
                         #p is point where acf increased significantly, it can be any points but in our graph we get increasing trend at 0.


# In[371]:


n_test = 32
    # model configs
cfg_list = sarima_configs()
    # grid search
scores = grid_search(centre_likes_ts, cfg_list, n_test)
print('done')
    # list top 3 configs
for cfg, error in scores[:3]:
    print(cfg, error)


# In[372]:


sarima = SARIMAX(centre_likes_ts, order=(11,1,10), seasonal_order=(2,1,3,12))
sarima_predictions = sarima.fit().predict()                            #Links to get SARIMAX parameters :- https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
                                                                       #https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
                                    


# In[373]:


plt.figure(figsize=(16,4))
plt.plot(centre_likes_ts, label="Actual")
plt.plot(sarima_predictions, label="Predicted")
plt.title('No of centre likes', fontsize=20)
#plt.ylabel('Sales', fontsize=16)
plt.legend()


# In[374]:


print('RMSE OF SARIMA of centre Likes count: %.4f'%np.sqrt(sum((centre_likes_ts-sarima_predictions)**2)/(len(centre_likes_ts))))


# In[375]:


#LSTM model for left likes count for lookback=1


# In[440]:


dataset=left_likes_ts


# In[441]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1,1))
train_size = int(len(dataset) * 0.67)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[378]:


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):   #The function takes two arguments: the dataset, which is a NumPy array that we want to convert into a dataset, 
  #and the look_back, which is the number of previous time steps to use as input variables to predict the next time period — in this case defaulted to 1.
#This default will create a dataset where X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[379]:


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

bayesian_opt_tuner.search(train, train,epochs=75,
     #validation_data=(X_test, y_test)
     validation_split=0.2,verbose=1)


bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)
model = bayes_opt_model_best_model[0]
bayesian_opt_tuner.results_summary()
bayesian_opt_tuner.search_space_summary()


# In[442]:


look_back = 1
trainX, trainY = create_dataset(train, look_back) #converting as above mentioned function for training and testing.
testX, testY = create_dataset(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))     


# In[443]:


from keras import backend as K


# In[444]:


model = Sequential()
model.add(LSTM(units=4, input_shape=(1, look_back)))#4 are the hidden LSTM blocks and 1 input layer as visible.
#model.add(Dense(units=4))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#K.set_value(model.optimizer.learning_rate, 0.001)
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;


# In[383]:


model.optimizer.get_config()


# In[445]:


model.summary()


# In[446]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[447]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[448]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score RMSE OF LEFT Likes : %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score RMSE OF LEFT Likes: %.2f RMSE' % (testScore))


# In[449]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)         
trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[450]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
plt.show()


# In[390]:


#LSTM Model for right likes count, lookback=1


# In[484]:


dataset=right_likes_ts


# In[485]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1,1))
train_size = int(len(dataset) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[393]:


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

bayesian_opt_tuner.search(train, train,epochs=150,
     #validation_data=(X_test, y_test)
     validation_split=0.2,verbose=1)


bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)
model = bayes_opt_model_best_model[0]
bayesian_opt_tuner.results_summary()
bayesian_opt_tuner.search_space_summary()


# In[486]:


look_back = 1
trainX, trainY = create_dataset(train, look_back) #converting as above mentioned function for training and testing.
testX, testY = create_dataset(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))     


# In[487]:


model = Sequential()
model.add(LSTM(units=4, input_shape=(1, look_back)))#4 are the hidden LSTM blocks and 1 input layer as visible.
model.add(Dense(units=4))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#K.set_value(model.optimizer.learning_rate, 0.01)
model.fit(trainX, trainY, epochs=75, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;


# In[466]:


model.optimizer.get_config()


# In[467]:


model.summary()


# In[488]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[489]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[490]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score RMSE OF right Likes : %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score RMSE OF right Likes: %.2f RMSE' % (testScore))


# In[491]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)         
trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[492]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
plt.show()


# In[403]:


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


# In[406]:


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
model.add(LSTM(units=8, input_shape=(1, look_back)))#4 are the hidden LSTM blocks and 1 input layer as visible.
model.add(Dense(units=2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
K.set_value(model.optimizer.learning_rate, 0.001)
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;


# In[497]:


model.optimizer.get_config()


# In[498]:


model.summary()


# In[499]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[500]:


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])              #Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[501]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score RMSE OF centre Likes : %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score RMSE OF centre Likes: %.2f RMSE' % (testScore))


# In[502]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)         
trainPredictPlot[:, :] = numpy.nan                                              #Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[503]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot) #red is training data, green is testing data, blue is orginal dataset.
plt.plot(testPredictPlot)  #https://www.applause.com/blog/training-data-validation-data-vs-test-data(differences between training data, test dataset)
plt.show()


# In[416]:


#LSTM forecasting for left likes for feedback=2


# In[417]:


dataset1=left_likes_ts


# In[418]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(dataset1.values.reshape(-1,1))
train_size = int(len(dataset1) * 0.67)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset1) - train_size
train, test = dataset1[0:train_size,:], dataset1[train_size:len(dataset1),:]
print(len(train), len(test))


# In[419]:


for i in range(2,3):
    trainX, trainY = create_dataset(train, i) #converting as above mentioned function for training and testing.
    testX, testY = create_dataset(test, i)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, i))) #4 are the hidden LSTM blocks and 1 input layer as visible.
    #model.add(Dense(units=4))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;
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


# In[420]:


#LSTM forecasting for right likes , for feedeback=2;


# In[421]:


dataset2=right_likes_ts


# In[422]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset2 = scaler.fit_transform(dataset2.values.reshape(-1,1))
train_size = int(len(dataset2) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset2) - train_size
train, test = dataset2[0:train_size,:], dataset2[train_size:len(dataset2),:]
print(len(train), len(test))


# In[423]:


for i in range(2,3):
    trainX, trainY = create_dataset(train, i) #converting as above mentioned function for training and testing.
    testX, testY = create_dataset(test, i)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, i))) #4 are the hidden LSTM blocks and 1 input layer as visible.
    model.add(Dense(units=4))
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


# In[424]:


#LSTM for centre likes for feedback=2


# In[425]:


dataset3=centre_likes_ts


# In[426]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset3 = scaler.fit_transform(dataset3.values.reshape(-1,1))
train_size = int(len(dataset3) * 0.8)   #splitting the dataset into 67 and 33 percent.
test_size = len(dataset3) - train_size
train, test = dataset3[0:train_size,:], dataset3[train_size:len(dataset3),:]
print(len(train), len(test))


# In[427]:


for i in range(2,3):
    trainX, trainY = create_dataset(train, i) #converting as above mentioned function for training and testing.
    testX, testY = create_dataset(test, i)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(2, input_shape=(1, i))) #4 are the hidden LSTM blocks and 1 input layer as visible.
    model.add(Dense(units=2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) #training the data with epochs=100,batch size=1;
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


# In[428]:


#Multistep forecasting for left likes


# In[510]:


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


# In[513]:


# load dataset
series = left_likes_ts
# configure
n_lag = 2
n_seq = 6
n_test = 53
n_epochs = 100
n_batch = 1
n_neurons = 8
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


# In[514]:


model.summary()


# In[432]:


#multistep forecasting for right likes


# In[519]:


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
    model.add(Dense(n_neurons))
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


# In[522]:


# load dataset
series = right_likes_ts
# configure
n_lag = 2
n_seq = 6
n_test = 31
n_epochs = 75
n_batch = 1
n_neurons = 4
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


# In[523]:


model.summary()


# In[436]:


#multistep forecasting for centre likes


# In[437]:


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
    model.add(Dense(n_neurons))
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


# In[438]:


# load dataset
series = centre_likes_ts
# configure
n_lag = 2
n_seq = 6
n_test = 32
n_epochs = 75
n_batch = 1
n_neurons = 2
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


# In[439]:


model.summary()

