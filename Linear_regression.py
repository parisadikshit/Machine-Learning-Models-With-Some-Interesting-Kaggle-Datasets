#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# ### What's regresssion,what's its use:

# 
# (1)It is to find the relationship among the variables(between inputs and the outputs).
# (2)It is used to predict the response using new set of data knowing how old data performed.
# (3)Inputs(predictors) being independent variables and output(responses) being dependent variable.
# 

# Linear regression is most commmonly used type of regression where it's general equation is y=mx+b where m and b being the parameters that we control and Xs are the set of inputs and Ys are the set of oututs.
# If there are multiple inputs then it'll be easier to represent them as X={x1,x2,x3,...xn} where n are number of predictors given (inputs given) and the multiple outputs as Y={a0 + a1x1 + a2x2 + a3x3+...anxn + E) where E is the random error.

# We can calculate the estimators or the weights for a model by linear regression.For a model,f(x)=a0 + a1x1+ a2x2+ ...anxn,
# a0,a1,a2,...an are the estimators or weights for respective inputs.
# f(x) is called as the estimator response while this is expected to be better if closer to the y.
# The estimated response f(xi) for observations where i=1,2,3..n has to be as close as possible to the yi so the quantity (yi-f(xi)) is calcualted which is called as residual.
# We are interested to find out the SSR=(sum of squares of residuals) cause if we minimise the SSR,we are more likely to get the best weights for the model.

# The coefficient of determination (R^2) has to be 1 to get SSR=0 and so that the line can perfectly fit the data.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import seaborn as sns


# In[2]:


boston_dataset = load_boston()


# In[3]:


boston_data = pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
#boston_datasets.feature_names has the all the features


# In[4]:


boston_data.head()


# In[5]:


boston_data.tail()


# In[6]:


boston_data.shape


# In[7]:


boston_data.columns


# In[8]:


boston_data['MEDV']=boston_dataset.target


# In[9]:


boston_data.head()


# CRIM = | ZN = | INDUS = | CHAS = | NOX = | RM = | AGE = | DIS = | RAD = | TAX = | PTRATIO = | B = | LSTAT = | MEDV = |

# In[10]:


boston_data.isnull().sum()


# In[11]:


boston_data.describe().round(2)


# In[12]:


sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(10,8)})


# In[13]:


#plotting distribution of target values
sns.distplot(boston_data['MEDV'],
            bins=30,
           
           )
plt.show()


# In[14]:


boston_data[['AGE','CRIM','TAX']].describe().round(2)


# In[15]:


#
boston_data.hist(column='MEDV',bins=30)
plt.show()


# In[16]:


boston_data.hist(column='RM',bins=20)
''''It looks like the distribution of number of average rooms is normal distributed cause mean and median are so close to one another'''


# In[17]:


boston_data.hist(column='LSTAT',bins=20)
plt.show()


# # Correlation Matrix

# #### Correlation matrix is useful to find the linear relationship between the variables 

# In[18]:


correlation_matrix = boston_data.corr().round(2)


# In[19]:


sns.heatmap(data=correlation_matrix,annot=True)


# ###### The correlation coefficient ranges from -1 to 1. If the value is close to 1, it means that there is a strong positive correlation between the two variables. When it is close to -1, the variables have a strong negative correlation. 

# ### So,value near 1 indicates strongest positive correlation between variables while value near to -1 indicates strongest negative correlation between variables.

# ### It is important to find and select those feature variables which are highly correlated with the target variable for a well fitted linear model.

# In[ ]:




