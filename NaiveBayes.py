#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes On Titanic Dataset and Heart Dataset

# In[1]:


import pandas as pd


# # Titanic Disaster Survival Prediction Dataset:

# In[2]:


titanic = pd.read_csv('./kaggle/titanic.csv')


# In[3]:


titanic.head()


# In[4]:


from sklearn.metrics import accuracy_score


# In[5]:


titanic.info()


# In[6]:


titanic.drop(['PassengerId','Name', 'SibSp', 'Parch', 'Ticket', 'Embarked', 'Cabin'], axis = 1, inplace = True)


# In[7]:


titanic.Sex


# In[8]:


titanic.head()


# In[9]:


titanic['Age'].isnull()


# In[10]:


titanic.tail()


# In[11]:


titanic.columns


# In[12]:


titanic.Age.fillna(titanic['Age'].mean().round(2), inplace=True)


# In[13]:


titanic.tail()


# In[14]:


titanic.columns


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


target = titanic['Survived']


# In[17]:


features = titanic.drop('Survived', axis=1)


# In[18]:


features.head()


# In[19]:


target


# In[20]:


dummies = pd.get_dummies(features.Sex)


# In[21]:


features = pd.concat([features, dummies], axis=1)


# In[22]:


features


# In[23]:


features.drop('Sex', axis=1, inplace=True)


# In[24]:


features


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2)


# In[58]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[59]:


from sklearn.naive_bayes import GaussianNB


# In[60]:


model = GaussianNB()


# In[61]:


model.fit(X_train, y_train)


# In[62]:


prediction = model.predict(X_test)


# In[63]:



model.score(X_test,y_test)


# In[64]:


# also the last 20 examples being in test set and after prediction --comparing


# In[65]:


y_test[:20]


# In[66]:


prediction[:20]


# In[67]:


prediction_probability = model.predict_proba(X_test)


# In[68]:


prediction_probability


# # Heart Attack Prediction Dataset

# In[69]:


heart = pd.read_csv('./kaggle/heart.csv')


# In[70]:


heart.head()


# In[71]:


heart.info()


# In[72]:


y_h = heart.target


# In[73]:


X_h = heart.drop('target', axis=1)


# In[74]:


X_h


# In[95]:


X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2)


# In[96]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_h = scaler.fit_transform(X_train_h)
X_test_h = scaler.transform(X_test_h)


# In[97]:


heart_model = GaussianNB()


# In[98]:


heart_model.fit(X_train_h, y_train_h)


# In[99]:


prediction_h = heart_model.predict(X_test_h)


# In[100]:


# also the last 20 examples being in test set and after prediction --comparing


# In[101]:


y_test_h[:20] #if run for few more times the accuracy is being achieved


# In[102]:


prediction_h[:20]


# In[103]:


heart_model.score(X_test_h, y_test_h)


# In[84]:


prediction_probability_h = heart_model.predict_proba(X_test_h)


# In[85]:


prediction_probability_h


# In[ ]:




