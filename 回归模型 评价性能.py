# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 09:04:00 2019

@author: 92156
"""

#问题三  结合支持向量机回归.py
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd 
from sklearn.svm import SVR

d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhino2"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

#导入guokong
d2=d1["NO2_y"]
#"CO_y","NO2_y","SO2_y","O3_y"]
d2=np.array(d2)
d2_test, d21_test = train_test_split(d2, random_state = 33, test_size = 0.25)
d2_test=np.array(d2_test).reshape(-1,1)
d21_test=np.array(d21_test).reshape(-1,1)
d2_x=StandardScaler()
d2_test=d2_x.fit_transform(d2_test)
d21_test=d2_x.transform(d21_test)

#zikong
import numpy 
d3=d1["NO2_x"]
d3_train, d31_train = train_test_split(d3, random_state = 33, test_size = 0.25)
d3_train=np.array(d3_train).reshape(-1,1)
d31_train=np.array(d31_train).reshape(-1,1)
d3_x=StandardScaler()
d3_train=d3_x.fit_transform(d3_train)
d31_train=d3_x.transform(d31_train)
a = numpy.array(rbf_svr_y_predict)

b = numpy.array(d31_train).reshape(880,)
qq=a+b


#explained_variance_score(解释方差分)
from sklearn.metrics import explained_variance_score
explained_variance_score(qq, d21_test) 
#这个指标用来衡量我们模型对数据集波动的解释程度，如果取值为1时，模型就完美，越小效果就越差。
# Mean absolute error（平均绝对误差）
from sklearn.metrics import mean_absolute_error
mean_absolute_error(qq, d21_test)
#给定数据点的平均绝对误差，一般来说取值越小，模型的拟合效果就越好。
#Mean squared error（均方误差）
from sklearn.metrics import mean_squared_error
mean_squared_error(qq, d21_test)
#Median absolute error（中位数绝对误差）
from sklearn.metrics import median_absolute_error
median_absolute_error(qq, d21_test)
#中位数绝对误差适用于包含异常值的数据的衡量
#R² score（决定系数、R方）
from sklearn.metrics import r2_score
r2_score(qq, d21_test, multioutput='variance_weighted')
#其值越接近1，则变量的解释程度就越高，其值越接近0，其解释程度就越弱。


#交叉验证
from sklearn.model_selection import cross_val_score
cross_val_score(rbf_svr, X_train, Y_train,cv=5,scoring='neg_mean_absolute_error')