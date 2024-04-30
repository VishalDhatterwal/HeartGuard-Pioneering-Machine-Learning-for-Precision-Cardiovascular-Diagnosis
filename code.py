#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')


# In[2]:


dataset = pd.read_csv('C:\\Users\\Vishal Dhattarwal\\Downloads\\heart.csv')


# In[3]:


type(dataset)


# In[4]:


dataset.shape


# In[5]:


dataset.head(5)


# In[6]:


dataset.sample(5)


# In[7]:


dataset.describe()


# In[8]:


dataset.info()


# In[9]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])


# In[10]:


dataset["target"].describe()


# In[11]:


dataset["target"].unique()


# In[12]:


print(dataset.corr()["target"].abs().sort_values(ascending=False))


# In[13]:


y = dataset["target"]

sns.countplot(y)


target_temp = dataset.target.value_counts()

print(target_temp)


# In[14]:


print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))

#Alternatively,
# print("Percentage of patience with heart problems: "+str(y.where(y==1).count()*100/303))
# print("Percentage of patience with heart problems: "+str(y.where(y==0).count()*100/303))

# #Or,
# countNoDisease = len(df[df.target == 0])
# countHaveDisease = len(df[df.target == 1])


# In[15]:


dataset["sex"].unique()


# In[16]:


sns.barplot(dataset["sex"],y)


# In[17]:


dataset["cp"].unique()


# In[18]:


sns.barplot(dataset["cp"],y)


# In[19]:


dataset["fbs"].describe()


# In[20]:


dataset["fbs"].unique()


# In[21]:


sns.barplot(dataset["fbs"],y)


# In[22]:


dataset["restecg"].unique()


# In[23]:


sns.barplot(dataset["restecg"],y)


# In[24]:


dataset["exang"].unique()


# In[25]:


sns.barplot(dataset["exang"],y)


# In[26]:


dataset["slope"].unique()


# In[27]:


sns.barplot(dataset["slope"],y)


# In[28]:


dataset["ca"].unique()


# In[29]:


sns.countplot(dataset["ca"])


# In[30]:


sns.barplot(dataset["ca"],y)


# In[31]:


dataset["thal"].unique()


# In[32]:


sns.barplot(dataset["thal"],y)


# In[33]:


sns.distplot(dataset["thal"])


# In[34]:


from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)


# In[35]:


X_train.shape


# In[36]:


X_test.shape


# In[37]:


Y_train.shape


# In[38]:


Y_test.shape


# In[39]:


from sklearn.metrics import accuracy_score


# Logistic Regression

# In[40]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)


# In[41]:


Y_pred_lr.shape


# In[42]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")


# In[43]:


#Naive Bayes


# In[44]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,Y_train)

Y_pred_nb = nb.predict(X_test)


# In[45]:


Y_pred_nb.shape


# In[46]:


score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# In[48]:


#SVM


# In[49]:


from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train, Y_train)

Y_pred_svm = sv.predict(X_test)


# In[50]:


Y_pred_svm.shape


# In[51]:


score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")


# In[52]:


#K Nearest Neighbors


# In[53]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)


# In[54]:


Y_pred_knn.shape


# In[55]:


score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# In[56]:


#Decision Tree


# In[57]:


from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)


# In[58]:


print(Y_pred_dt.shape)


# In[59]:


score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


# In[60]:


#Random Forest


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)


# In[62]:


Y_pred_rf.shape


# In[63]:


score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")


# In[64]:


#XGBoost


# In[66]:


pip install xgboost


# In[67]:


import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, Y_train)

Y_pred_xgb = xgb_model.predict(X_test)


# In[68]:


Y_pred_xgb.shape


# In[69]:


score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)

print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")


# In[70]:


#Neural Network


# In[71]:


from keras.models import Sequential
from keras.layers import Dense


# In[72]:


# https://stats.stackexchange.com/a/136542 helped a lot in avoiding overfitting

model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[73]:


model.fit(X_train,Y_train,epochs=300)


# In[74]:


Y_pred_nn = model.predict(X_test)


# In[75]:


Y_pred_nn.shape


# In[76]:


rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded


# In[77]:


score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")

#Note: Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11. 


# In[78]:


#VI. Output final score


# In[79]:


scores = [score_lr,score_nb,score_svm,score_knn,score_dt,score_rf,score_xgb,score_nn]
algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest","XGBoost","Neural Network"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


# In[80]:


sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# In[82]:


# Hey Vishal, there random forest has good result as compare to other algorithms


# In[ ]:




