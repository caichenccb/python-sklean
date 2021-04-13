# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:51:29 2020

@author: 92156
"""

import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import  stats
import warnings
warnings.filterwarnings("ignore")
import sys

train_data_file="D:\天池数据\工业预测\zhengqi_train.txt"
text_data_file="D:\天池数据\工业预测\zhengqi_test.csv"
train_data=pd.read_csv(train_data_file,sep="\t",encoding="unicode_escape")
text_data=pd.read_csv(text_data_file,encoding="gbk")


text_data.info()
train_data.info()


train_data.describe()


text_data.describe()

train_data.head()
text_data.head()

fig=plt.figure(figsize=(5,9))
sns.boxplot(train_data["V0"],orient="v",width=0.5)
plt.show()

column=train_data.columns.tolist()[:39]
fig=plt.figure(figsize=(80,60),dpi=75)
for i in range(38):
    plt.subplot(7,8,i+1) #13行3列子图
    sns.boxplot(train_data[column[i]],orient="v",width=0.5) # 箱形图
    plt.ylabel(column[i],fontsize=36)
plt.show()

# 岭回归

#function to detect outliers based on the predictions of model 
def find_outliers(model,X,y,sigma=3):
    #predict. y value using model 
    try: 
        y_pred=pd.Series(model.predict(X),index=y.index)
    #if predicting fails ,try fitting the model first \
    except:
        model.fit(X,y)
        y_pred=pd.Series(model.predict(X),index=y.index)
    #calculat residuale betweenw the model prediction and true y values
    resid=y-y_pred
    mean_resid=resid.mean()
    std_resid=resid.std()
    #calculate z statistic , define outliers to be where |z| > sigma
    z=(resid-mean_resid)/std_resid
    outliers=z[abs(z)>sigma].index
    
    #print and plot the results 
    
    print("R2=",model.score(X,y))
    print("msc=",mean_squared_error(y,y_pred))
    print("------------------------")
    
    print("mean of residuals:",mean_resid)
    print("std of residuals :",std_resid)
    print("------------------------")
    
    
    print(len(outliers),"outliers:")
    print(outliers.tolist())
    
    plt.figure(figsize=(15,5))
    ax_131=plt.subplot(1,3,1)
    plt.plot(y.loc[outliers],y_pred.loc[outliers],"ro")
    plt.legend(["Accepted","Outlier"])
    plt.xlabel("y")
    plt.ylabel("y_pred")
    
    ax_132=plt.subplot(1,3,2)
    plt.plot(y,y-y_pred,".")
    plt.plot(y.loc[outliers],y.loc[outliers]-y_pred.loc[outliers],"ro")
    plt.legend(["Accepted","Outlier"])
    plt.xlabel("y")
    plt.ylabel("y-y_pred")
    
    ax_133=plt.subplot(1,3,3)
    z.plot.hist(bins=50,ax=ax_133)
    z.loc[outliers].plot.hist(color="r",bins=50,ax=ax_133)
    plt.legend(["Accepted","Outlier"])
    plt.xlabel("z")
    
    return outliers


from sklearn.linear_model import  Ridge
from sklearn.metrics import  mean_squared_error
X_train=train_data.iloc[:,0:-1]
y_train=train_data.iloc[:,-1]
outliers=find_outliers(Ridge(),X_train,y_train)



