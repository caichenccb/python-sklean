# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:24:28 2019

@author: 92156
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("疫情.xlsx",encoding="gbk")
y=np.array(d1["trade_date"])
X=np.array(d1["close"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
#%画图 PM2.5 /15  00：14
plt.plot(rbf_svr_y_predict[:100,],c="r",label="NO2预测")
plt.plot(y_test[:100,],c="b",label='NO2')
plt.legend()
plt.ylabel("real and predicted value") 
plt.title("regression result comparison")