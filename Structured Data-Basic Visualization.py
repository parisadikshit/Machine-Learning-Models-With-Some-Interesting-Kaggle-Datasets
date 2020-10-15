#!/usr/bin/env python
# coding: utf-8

# In[1]:


#loading modules
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('kaggle/mall_customers/Mall_Customers.csv')


# In[3]:


data.head()


# In[4]:


data.drop(['CustomerID'],axis=1,inplace=True)
data.mean()


# In[5]:


data.std()


# In[6]:


pd.plotting.andrews_curves(data,'Genre') #still a doubt


# In[7]:


pd.plotting.scatter_matrix(data,alpha=0.2)


# In[8]:


ax = sns.relplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=data)


# In[9]:



ax = sns.relplot(data=data)


# In[10]:


annual_income = data['Annual Income (k$)']
ax = sns.displot(
    annual_income,
    bins=50

)


# In[11]:


ax = sns.displot(
    annual_income,
    

)
ax.set(xlabel='Annual Income (k$)',ylabel='Frequency')


# In[12]:


data.head()


# In[ ]:




