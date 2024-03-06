#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv('UserBehavior.csv',header = None)
data.columns = ['userId','goodsId','categoryId','behaviorType','timeStamp']
data.head(5)


# In[2]:


print(pd.isnull(data["userId"]).value_counts())
print(pd.isnull(data["goodsId"]).value_counts())
print(pd.isnull(data["categoryId"]).value_counts())
print(pd.isnull(data["behaviorType"]).value_counts())
print(pd.isnull(data["timeStamp"]).value_counts())


# In[3]:


data.info()


# In[4]:


print(data["behaviorType"].value_counts())


# In[5]:


print(data.duplicated().value_counts())
data = data.drop_duplicates()
print(data.duplicated().value_counts())


# In[6]:


data.info()


# In[7]:


import time
data = data[data["timeStamp"]>0]
data.loc[:,'timeStamp']=data['timeStamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))


# In[8]:


dic = {'userId':str,'goodsId':str,'categoryId':str,'behaviorType':str,'timeStamp':'datetime64[ns]'}
data = data.astype(dic)
data.info()


# In[9]:


data=data[(data["timeStamp"]>='2017-11-25')&(data["timeStamp"]<'2017-12-04')]
print(data["timeStamp"].value_counts())


# In[10]:


data.describe()


# In[11]:


data.loc[:,'date']=data['timeStamp'].dt.date
data.loc[:,'time']=data['timeStamp'].dt.time
data.head(5)


# In[12]:


smalldata = data.sample(n = 10000000, axis = 0)
smalldata.describe()


# In[13]:


smalldata.to_csv('UserBehaviorSmall.csv.bz2',index=False,compression={'method':'bz2'})


# In[14]:


data.to_csv('UserBehavior.csv.bz2',index=False,compression={'method':'bz2'})


# In[15]:


smalldata

