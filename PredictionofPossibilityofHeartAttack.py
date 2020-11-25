#!/usr/bin/env python
# coding: utf-8

# # Predecting the possibility of Heart Attack

# In[1]:


#necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


sns.set(color_codes=True)
sns.set_theme(font_scale=10.5)
sns.set(rc={'figure.figsize':(15,15)})


# In[3]:


from sklearn.preprocessing import StandardScaler


# In[4]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


from sklearn.linear_model import LogisticRegression


# In[7]:


heart = pd.read_csv('./kaggle/heart.csv')


# In[8]:


heart.head()


# In[9]:


#1.Age
#2.Sex
#3.chest pain type (4 values)
#4.resting blood pressure
#5.serum cholestoral in mg/dl
#6.fasting blood sugar > 120 mg/dl
#7.resting electrocardiographic results (values 0,1,2)
#8.maximum heart rate achieved
#9.exercise induced angina
#10.oldpeak = ST depression induced by exercise relative to rest
#11.the slope of the peak exercise ST segment
#12.number of major vessels (0-3) colored by flourosopy
#13.thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
#14.target: 0= less chance of heart attack 1= more chance of heart attack


# In[10]:


heart.info()


# In[11]:


print("Maximum and Minimum Cholesterol from Data respectively:",(heart['chol'].max(),heart['chol'].min()))


# In[12]:


female = heart['sex']==0


# In[13]:


ax = sns.displot(heart,
                 x=heart['chol'])


# In[14]:


ax = sns.displot(
    heart,
    x='chol',
    y='target',
    color='red'
)
#persons with cholesterol above 400 having higher chances of heart attack


# In[15]:


independent_variables = heart.iloc[:,0:13]


# In[16]:


independent_variables


# In[17]:


dependent_variable = heart.iloc[:,13:]


# In[18]:


corrmatrix = heart.corr()


# In[19]:


corrmatrix


# In[20]:


# negatively correlated and positively correlated features with target can be seen clearly through correlation matrix


# In[21]:


plt.figure(figsize=(18,18))
#plotting the heat map
p = sns.heatmap(corrmatrix, annot=True, cmap = 'RdYlGn')
plt.title("Correlation between the features and the target")


# In[22]:


#just for seeing how are these 3 features ranging with each other
chol_age_sex ={
 'chol':heart['chol'],
    'age':heart['age'],
    'sex':heart['sex'],
    'target':heart['target']

}


# In[23]:


chol_age_sex = pd.DataFrame(chol_age_sex,columns=['chol','age','sex','target'])


# In[24]:


chol_age_sex


# In[25]:


chol_age_sex_corrmat = chol_age_sex.corr()


# In[26]:


plt.figure(figsize=(5,5))
sns.heatmap(chol_age_sex_corrmat,annot=True, cmap='RdYlGn')


# In[27]:


y = heart['target']
X = heart.drop('target',axis=1)


# In[28]:


X


# In[29]:


y = heart['target']


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[32]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# # Logistic Regression

# In[33]:


clf = LogisticRegression()


# In[34]:


clf.fit(X_train,y_train)


# In[35]:


prediction = clf.predict(X_test)


# In[36]:


confus_matrix = confusion_matrix(y_test, prediction)
clf_report = classification_report(y_test, prediction)
accur_score = accuracy_score(y_test, prediction)
print("**Accuracy acchieved using Logistic Regression:", accur_score*100)
print("**Classification report:")
print(clf_report)


# # KNeighborsClassifier

# In[37]:


#Applying KNN classifier
from sklearn.neighbors import KNeighborsClassifier


# In[38]:


knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
prediction_knn = knn.predict(X_test)
confus_matrix_knn = confusion_matrix(prediction_knn, y_test)
acc_score_knn = accuracy_score(prediction_knn, y_test)
clf_report_knn = classification_report(prediction_knn, y_test)
print("**Accuracy acchieved using KNeigborsClassifier:", acc_score_knn*100)
print("**Classification report:")
print(clf_report_knn)


# In[39]:


print("**Accuracy acchieved using KNeigborsClassifier when 11 neighbors tested:", acc_score_knn*100)


# In[40]:


from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors':np.arange(1,50)}
knn_cv= GridSearchCV(knn,param_grid,cv=10)
knn_cv.fit(X,y)


# In[41]:


knn_cv.best_score_


# In[42]:


knn_cv.best_params_


# In[43]:


knn = KNeighborsClassifier(n_neighbors=29)
knn.fit(X_train, y_train)
prediction_knn = knn.predict(X_test)
confus_matrix_knn = confusion_matrix(prediction_knn, y_test)
acc_score_knn = accuracy_score(prediction_knn, y_test)
clf_report_knn = classification_report(prediction_knn, y_test)
print("**Accuracy acchieved using KNeigborsClassifier:", acc_score_knn*100)
print("**Classification report:")
print(clf_report_knn)


# In[44]:


print("**Accuracy acchieved using KNeigborsClassifier when 29 neighbors tested:", acc_score_knn*100)


# # ChiSquare (For Selecting Best Features)

# In[45]:


# just for observation for deciding which are best features describing prediction
#chi2
from sklearn.feature_selection import SelectKBest, chi2


# In[46]:


best_features = SelectKBest(score_func = chi2 , k = 5)


# In[47]:


fit = best_features.fit(X,y)


# In[48]:


fitscoreDataframe = pd.DataFrame(fit.scores_)


# In[49]:


featureDataframe = pd.DataFrame(X.columns)


# In[50]:


featurescores = pd.concat([featureDataframe,fitscoreDataframe],axis=1)


# In[51]:


featurescores.columns = ['Specs','Score']


# In[52]:


featurescores


# In[53]:


featurescores.nlargest(5,'Score')


# In[ ]:




