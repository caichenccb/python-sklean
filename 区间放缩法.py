# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 23:26:49 2019

@author: 92156
"""

from sklearn.preprocessing import MinMaxScaler
 
#区间缩放，返回值为缩放到[0, 1]区间的数据
MinMaxScaler().fit_transform(iris.data)


#X'=(x-MIN)/(max-min)