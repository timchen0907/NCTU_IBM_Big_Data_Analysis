# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:14:58 2021

@author: Tim Chen
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from chefboost import Chefboost as chef
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
data=pd.read_csv("online_shoppers_intention.csv",encoding="utf-8-sig")
dummy_ls=list(data.columns[-8:-2])
for d in dummy_ls:
    tmp_df=pd.get_dummies(data[d],prefix=d+"_",drop_first=True)
    data=pd.concat([data,tmp_df],axis=1)
    del data[d]
    
data["weekend"]=np.where(data["Weekend"]==True,1,0)
del data["Weekend"]
data["revenue"]=np.where(data["Revenue"]==True,1,0)
del data["Revenue"]

# data.to_csv("paper_study3.csv",encoding="utf-8-sig")
x=data.drop(columns=["revenue"])
y=data["revenue"]
data["revenue"].value_counts()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1207)

train = pd.concat([x_train, y_train], axis = 1)
valid = pd.concat([x_test, y_test], axis = 1)
re = train[train["revenue"] == 1]
not_re = train[train["revenue"] == 0]
# low = train[train["interest_level"] == 'low']
df_re = re.sample(len(not_re),replace=True)

#
train_ros = pd.concat([df_re,not_re],axis = 0)
train_ros = train_ros.sample(frac=1)
# train_ros.to_csv("train_oversample.csv",encoding="utf-8-sig")
# valid.to_csv("test.csv",encoding="utf-8-sig")
X_train_ros = train_ros.drop(columns=["revenue"])
Y_train_ros = train_ros['revenue']
Y_train_ros.value_counts()
#%%
clf=GradientBoostingClassifier(random_state=1130,max_depth=2 )
clf.fit(x_train, y_train)
train_pred=clf.predict(x_train)
print("Accuracy:",metrics.accuracy_score(y_train, train_pred))
test_pred=clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, test_pred))
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt='d',cmap='YlGnBu')
plt.ylabel("True Value", color = 'r')
plt.xlabel("Predicted Value", color = 'r')
tn=cm[0][0]
tp=cm[1][1]
fn=cm[1][0]
fp=cm[0][1]

print("TPR:", tp/(tp+fn))
print("TNR:", tn/(tn+fp))
print("F1-score:",metrics.f1_score(y_test,test_pred))

#%%
train=pd.concat([x_train,y_train],axis=1)
config={"algorithm":"C4.5","enableParallelism":True, "num_cores":8}
train["Decision"]=np.where(train["revenue"]==1,"Yes","No")
del train["revenue"]
model=chef.fit(train,config)

chef_test_pred=[]
for i in tqdm(range(len(x_test))):
    pred=chef.predict(model,x_test.iloc[i])
    if pred=="No":
        chef_test_pred.append(0)
    else:
        chef_test_pred.append(1)

chef_cm=confusion_matrix(y_test,chef_test_pred)
sns.heatmap(chef_cm, annot=True, fmt='d',cmap='YlGnBu')
plt.ylabel("True Value", color = 'r')
plt.xlabel("Predicted Value", color = 'r')
tn=chef_cm[0][0]
tp=chef_cm[1][1]
fn=chef_cm[1][0]
fp=chef_cm[0][1]

print("TPR:", tp/(tp+fn))
print("TNR:", tn/(tn+fp))
print("F1-score:",metrics.f1_score(y_test,chef_test_pred))
print("Accuracy:",metrics.accuracy_score(y_test, chef_test_pred))

#%%
# oversample #
clf=GradientBoostingClassifier(random_state=1130,max_depth=2 )
clf.fit(X_train_ros, Y_train_ros)
train_pred=clf.predict(X_train_ros)
print("Accuracy:",metrics.accuracy_score(Y_train_ros, train_pred))
test_pred=clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, test_pred))
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt='d',cmap='YlGnBu')
plt.ylabel("True Value", color = 'r')
plt.xlabel("Predicted Value", color = 'r')
tn=cm[0][0]
tp=cm[1][1]
fn=cm[1][0]
fp=cm[0][1]

print("TPR:", tp/(tp+fn))
print("TNR:", tn/(tn+fp))
print("F1-score:",metrics.f1_score(y_test,test_pred))
#%%
train=pd.concat([X_train_ros,Y_train_ros],axis=1)
config={"algorithm":"C4.5","enableParallelism":True, "num_cores":8}
train["Decision"]=np.where(train["revenue"]==1,"Yes","No")
del train["revenue"]
model=chef.fit(train,config)

chef_test_pred=[]
for i in tqdm(range(len(x_test))):
    pred=chef.predict(model,x_test.iloc[i])
    if pred=="No":
        chef_test_pred.append(0)
    else:
        chef_test_pred.append(1)

chef_cm=confusion_matrix(y_test,chef_test_pred)
sns.heatmap(chef_cm, annot=True, fmt='d',cmap='YlGnBu')
plt.ylabel("True Value", color = 'r')
plt.xlabel("Predicted Value", color = 'r')
tn=chef_cm[0][0]
tp=chef_cm[1][1]
fn=chef_cm[1][0]
fp=chef_cm[0][1]

print("TPR:", tp/(tp+fn))
print("TNR:", tn/(tn+fp))
print("F1-score:",metrics.f1_score(y_test,chef_test_pred))
print("Accuracy:",metrics.accuracy_score(y_test, chef_test_pred))