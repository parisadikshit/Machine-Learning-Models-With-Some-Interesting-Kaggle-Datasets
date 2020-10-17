#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Continuation

# In[1]:


import pandas as pd
from sklearn.datasets import load_boston,load_wine


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(10,10)})


# In[4]:


boston_dataset = load_boston()


# In[5]:


boston_data = pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)


# In[6]:


boston_data['MEDV']=boston_dataset.target


# In[7]:


boston_data.head(5)


# # Feature Selection

# - Finding and training the model on relevant data becomes important because if trained on completely or partially irrelevent features,the model won't give accuaracy. 

# - selecting features wisely to train the model, we improve the accuaracy,reduce the overfitting and will not get misleaded by redundant data.

# In[8]:


boston_correlation = boston_data.corr().round(2)


# In[9]:


top_core_features = boston_correlation.index


# In[10]:


sns.heatmap(boston_correlation,annot=True,cmap='RdYlGn')


# In[11]:


sns.heatmap(boston_correlation[top_core_features].corr(),annot=True,cmap='RdYlGn')


# > - As LSTAT decreases the MEDV increases,as RM increases the MEDV increses too

# # Scatter Plots

# > - Scatter Plots are used to observe relationship between 2 variables.

# In[12]:


boston_data.plot(
    kind='scatter',
    x='RM',
    y='MEDV',
    color='green'
)


# > The price increases as the number of rooms increases.

# In[13]:


boston_data.plot(
    kind='scatter',
    x='LSTAT',
    y='MEDV',
    color='red',
    label=''
)


# > The prices decreases as the LSTAT increases <br> So,we know that the better feature to predict about MEDV with future data is this feature RM <br> So,the line equation becomes something like y= m*(RM) + c

# In[14]:


X = boston_data[['RM']]


# In[15]:


Y = boston_data['MEDV']


# In[16]:


X.shape


# In[17]:


Y.shape


# In[18]:


boston_data['MEDV'].describe()


# # Model Instantiation

# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


model = LinearRegression()


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)


# In[23]:


X_train.shape


# In[24]:


X_test.shape


# In[25]:


Y_train.shape


# In[26]:


Y_test.shape


# # Fitting the model

# In[27]:


model.fit(X_train,Y_train)


# # Estimating the Parameters

# In[28]:


model.intercept_


# In[29]:


model.intercept_.round(2)


# In[30]:


model.coef_


# In[31]:


model.coef_.round(2)


# Therefore, MEDV = -30.57 + 8.46*(RM)

# # Prediction

# In[32]:


y_test_prediction = model.predict(X_test)


# In[33]:


y_test_prediction.shape


# In[34]:


type(y_test_prediction)


# In[35]:


plt.scatter(
    X_test,
    Y_test,
    label='Testing The Model',

)
plt.plot(
    X_test,
    y_test_prediction,
    label='Prediction',
    linewidth=2,
)
plt.xlabel='RM'
plt.ylabel='MEDV'
plt.legend(loc='upper left')
plt.show()


# # Evaluating the Model

# In[36]:


residual = Y_test - y_test_prediction


# In[37]:


plt.scatter(X_test,residual)
plt.hlines(y=0,xmin=X_test.min(),xmax=X_test.max(),linestyle='--')
xlim=(4,9)
plt.show()


# In[38]:


residual[:5].round(3)


# In[39]:


(residual**2).mean()


# In[40]:


from sklearn.metrics import mean_squared_error


# In[41]:


mean_squared_error(Y_test,y_test_prediction)


# # R Squared

# In[42]:


model.score(X_test,Y_test)


# # Multivarient Linear Regression

# In[43]:


X2 = boston_data[['RM','LSTAT']]


# In[44]:


Y2 = boston_data['MEDV']


# In[45]:


X2_train,X2_test,Y2_train,Y2_test = train_test_split(X2,Y2,test_size=0.3,random_state=1)


# In[46]:


model2= LinearRegression()


# In[47]:


model2.fit(X2_train,Y2_train)


# In[48]:


model2.intercept_


# In[49]:


model2.coef_


# In[50]:


y2_test_prediction = model2.predict(X2_test)


# # Evaluating the model again

# # Mean Squared Error

# In[51]:


mean_squared_error(Y2_test,y2_test_prediction)


# In[52]:


model2.score(X2_test,Y2_test)

