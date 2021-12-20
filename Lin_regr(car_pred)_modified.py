#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np


# In[51]:


df = pd.read_csv('data.csv')


# In[52]:


#data preparation
df.columns = df.columns.str.replace(' ','_').str.lower()


# In[53]:


#not object data
df = df[list(df.dtypes[df.dtypes != object].index)]


# In[54]:


df_col_indexes = list(df.dtypes.index)


# In[55]:


#if foolish data replace by mean
for elem in df_col_indexes:
    if True in df[elem].isnull():
        df[elem] = df[elem].fillna(df[elem].mean())


# In[56]:


n = len(df)


# In[57]:


#split data
n_train = int(n*0.6)
n_val = int(n*0.2)
n_test = n - n_val - n_train


# In[58]:


n == n_train + n_val + n_test


# In[59]:


#random string order
mix = np.arange(n)
np.random.seed(2)
np.random.shuffle(mix)


# In[60]:


#splitted data
df_train = df.iloc[mix[:n_train]]
df_val = df.iloc[mix[n_train:n_train + n_val]]
df_test = df.iloc[mix[n_train + n_val:]]


# In[61]:


#new indexes for train
df_train = df_train.reset_index(drop = True)


# In[62]:


#values to log for train
val_train = np.log1p(df_train.msrp.values)


# In[63]:


#get rid of msrp for train
df_train = df_train.drop(['msrp'], axis=1)


# In[64]:


x = (df_train[list(df_train.dtypes.index)].values)
x = np.column_stack((np.ones(np.shape(x)[0]),x))


# In[65]:


#search proper weights
temp_matrix = (x.T).dot(x)
temp_matrix = np.linalg.inv(temp_matrix)
temp_matrix = temp_matrix.dot(x.T)

w = temp_matrix.dot(val_train)


# In[66]:


#new indexes for test
df_test = df_test.reset_index(drop = True)


# In[67]:


#values to log for test
val_test = np.log1p(df_test.msrp.values)


# In[68]:


#get rid of msrp for test
df_test = df_test.drop(['msrp'], axis=1)


# In[69]:


y = (df_test[list(df_test.dtypes.index)].values)
y = np.column_stack((np.ones(np.shape(y)[0]),y))


# In[70]:


#prediction
res = y.dot(w)


# In[71]:


#log err
err = 0

for i in range(len(res)):
    err = abs(res[i] - val_test[i])
    print(i, err)


# In[72]:


#exp err
err = 0

for i in range(len(res)):
    err = abs(np.expm1(res[i]) - np.expm1(val_test[i]))
    print(i, err)


# In[73]:


#mean log err
err_log = abs(res - val_test).mean()
err_log


# In[74]:


#mean exp err
err_exp = abs(np.expm1(res) - np.expm1(val_test)).mean()
err_exp


# In[ ]:




