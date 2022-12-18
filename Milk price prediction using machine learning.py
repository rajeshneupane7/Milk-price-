#!/usr/bin/env python
# coding: utf-8

# # Milk price prediction in US using historical dataset 

# In[227]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (20,15)
import seaborn as sns
import math
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU


# In[228]:


dataframe= pd.ExcelFile('/Users/rajesh/Downloads/SeriesReport-20221211034635_cdcc19.xlsx')


# In[229]:


dataframe=pd.read_excel(dataframe, 'BLS Data Series')


# In[230]:


dataframe


# In[231]:


dataframe['Period'] = dataframe['Period'].map(lambda x: x.lstrip('M').rstrip('M'))


# In[232]:


dataframe


# In[233]:


dataframe['Period']=dataframe['Period'].astype(str)


# In[234]:


dataframe['Year']=dataframe['Year'].astype(str)


# In[235]:


dataframe['Date'] = dataframe[['Year', 'Period']].agg('-'.join, axis=1)


# In[236]:


dataframe


# In[237]:


dataframe.index = pd.to_datetime(dataframe.Date)

dataframe.info()


# In[238]:


dataframe


# In[239]:


dataframe= dataframe.filter(items=['Value'], axis=1)


# In[240]:


dataframe


# In[241]:


dataframe.plot()


# In[242]:


len(dataframe)


# In[346]:




# Split
train = int(len(dataframe)*.8)

Xtrain, Xtest = dataframe.iloc[:train,:], dataframe.iloc[train:,:]


# In[347]:


Xtrain


# In[348]:


def make_dataset(data, window=1):
    X, y = [], []
    for i in range(len(data) - window -1):
        X.append(data.iloc[i:i + window,:])
        y.append(data.iloc[i + window,:])
    return np.array(X), np.array(y)


# In[349]:


Xtrain, ytrain = make_dataset(Xtrain, 5)
Xtest, ytest = make_dataset(Xtest, 5)


# In[ ]:





# In[331]:


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


# In[332]:


model=Sequential()
model.add(LSTM(60, return_sequences=True, input_shape= (5,1)))
model.add(LSTM(60))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit( Xtrain, ytrain,validation_split=0.2, epochs=100, verbose=1, callbacks=[stop_early], )


# In[333]:


prediction= model.predict(Xtest)


# In[334]:


mean_squared_error(ytest, prediction)


# In[ ]:





# In[297]:


plt.plot(ytest, label='Original Test')
plt.plot(prediction, label='Test Predictions')


# In[252]:


plt.plot(ytrain, label='Original')
plt.plot(model.predict(Xtrain), label='Train Predictions')
plt.legend()


# In[96]:


mean_squared_error(ytrain, model.predict(Xtrain))


# In[98]:



model=Sequential()
model.add(GRU(60,return_sequences=True,input_shape=(5,1)))
model.add(GRU(60))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(Xtrain,ytrain,validation_data=(Xtest,ytest),epochs=100,batch_size=1,verbose=1)


# In[99]:




plt.plot(ytest, label='Original Training')
plt.plot(model.predict(Xtest), label='Train Predictions')
plt.legend()


# In[ ]:




