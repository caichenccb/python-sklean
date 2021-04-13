# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:16:24 2019

@author: 92156
"""

from sklearn import linear_model    # Ridge岭回归,RidgeCV带有广义交叉验证的岭回归
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = pd.read_excel("工作簿1(2).xlsx")
alphas = 10**np.linspace(- 3, 3, 100)
X=data[["Aver pres","High pres","Low pres","Aver temp","High temp","Low temp","Aver RH","Min RH"]]
y = data['Rizhenhengduanliang']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
lr = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99],max_iter=5000).fit(X_train, y_train)
y_prediction = lr.predict(X_test)


