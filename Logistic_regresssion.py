#!/usr/bin/env python
# coding: utf-8

# # Logistic  Regression

# > A regression method which is used for solving the classification problems.
# It is based on the **logistic function**(Sigmoid function).While doing linear regressions, we changed the weights in relevance with the result that we got from mean square error but we can not use MSE at some instances. The loss function in logisitc regression is called as **log loss**.

# > Here we have loss function someyhing like :
# >> **Log loss = $ -ylog(\hat y) - (1-y)log(1-\hat y)    $**
# 
# we give $ \hat y $ as the sigmoid function i.e. $ \hat y = 1/(1+e^{-y})$

# >And the sigmoid function looks like:
#     >>    **$  \sigma(z) = 1/(1 + e^{-z}) $**
#   

# while doing linear regression, y = w.T*x + b, we put this y inside the sigmoid function to get $ \hat y $.
# While that is just for only one example. for m, the loss function will be like:
# >> $Loss function=  \sum\limits_{i=1}^{m}{-y_{i}log(\hat y_{i}) - (1-y_{i})log(1-\hat y_{i})}    $

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


# In[3]:


iris = load_iris()


# In[4]:


iris.target_names


# In[5]:


iris.feature_names


# In[6]:


data = pd.DataFrame(np.c_[iris['data'],iris['target']], columns = iris['feature_names']+['target'])


# In[7]:


data


# In[8]:


print(data)


# In[9]:


iris['data'][50:,]


# In[10]:


X = iris['data'][:,3:]


# In[11]:


y = iris['target']==2 #virginica


# In[12]:


y = y.astype(np.int)


# In[13]:


obj = LogisticRegression()


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[15]:


obj.fit(X_train,y_train)


# In[16]:


obj.predict(X_test)


# In[17]:


obj.predict([[1.6]])


# In[18]:


arr = np.linspace(0,3,100000).reshape(-1, 1)


# In[19]:


arr


# In[20]:


arr_proba = obj.predict_proba(arr)


# In[21]:


arr_proba


# In[22]:


plt.plot(arr, arr_proba[:,1], 'r-',label='virginica')
plt.title('Logistic Regression')
plt.ylabel('Probability')
plt.show()


# In[ ]:




