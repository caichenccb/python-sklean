# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:41:14 2019

@author: 92156
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame

#导入库


#cc6.to_csv("cc6.csv",encoding="gbk")
#防止输出中文乱码
cc1=pd.read_csv("cc1.csv")
cc2=pd.read_csv("cc2.csv")
cc3=pd.read_csv("cc3.csv")
cc4=pd.read_csv("cc4.csv")

#读表
cc1["id1"]=1
cc2["id2"]=2
cc3["id3"]=3
cc4["id4"]=4

#给表上标识
cc5=pd.merge(cc3,cc2,on=["spbm","dtime","je"],how="outer",suffixes=("_left","_right"))
#匹配出所有人的消费
cc6=pd.merge(cc3,cc1,on=["kh"],suffixes=("_left","_right"))
#匹配出本店会员
cc7=pd.merge(cc2,cc6,on=["spbm","dtime","je"],suffixes=("_left","_right"))   #本店会员
#匹配出本店活会员消费记录
cc8=cc5[cc5.id2.isin([2])&-cc5.id3.isin([3])]    #非本店会员
#检索出非本店会员消费


#cc9['csny'] = pd.to_datetime(cc9['csny'])
#
#import datetime as dt
#now_year =dt.datetime.today().year  #当前的年份
#cc9['age']=now_year-cc9.csny.dt.year
#
##计算年龄

def status(x):
 return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),
                      x.quantile(.75),x.mean(),x.max(),ss.idxmax(),x.mad(),x.var(),
                      x.std(),x.skew(),x.kurt()],index=['总数','最小值','最小值位置','25%分位数',
                    '中位数','75%分位数','均值','最大值','最大值位数','平均绝对偏差','方差','标准差','偏度','峰度'])
                    
#计算总数','最小值','最小值位置','25%分位数','中位数','75%分位数','均值','最大值','最大值位数','平均绝对偏差','方差','标准差','偏度','峰度'

    
    
    