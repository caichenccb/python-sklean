# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 14:39:58 2019

@author: 92156
"""

import pandas as pd
from pylab import *
import numpy as np 
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
from matplotlib import font_manager
import seaborn as sns
from numpy.random import randn
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

#points=np.arange(-5,5,0.01)
#xs,ys=np.meshgrid(points,points)
#ys
#z=np.sqrt(ys**2+xs**2)
#z
#plt.imshow(z,cmap=plt.cm.Pastel1);
#cm.的色条
#https://matplotlib.org/examples/color/colormaps_reference.html
#https://matplotlib.org/users/colormaps.html
#plt.colorbar()
#plt.title("xxxxx")
#from pylab import *
#
#def f(x,y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
#
#n = 10
#x = np.linspace(-3,3,4*n)
#y = np.linspace(-3,3,3*n)
#X,Y = np.meshgrid(x,y)
#imshow(f(X,Y)), show()

#随机多次漫步

#import numpy as np
#nwalks=5000;nsteps=1000;draws=np.random.randint(0,2,size=(nwalks,nsteps))
#steps=np.where(draws>0,1,-1)
#walks= steps.cumsum(1)
#walks

#data=randn(30).cumsum()
#plt.plot(data,"k--")
#plt.plot(data,"k-",drawstyle="steps-post")

#凡小于百分之1分位数和大于百分之99分位数的值将会被百分之1分位数和百分之99分位数替代：
#def cc(x,quantile=[0.01,0.99]):
##    """盖帽法处理异常值
##    Args：
##        x：pd.Series列，连续变量
##        quantile：指定盖帽法的上下分位数范围
##    """
#
## 生成分位数
#    Q01,Q99=x.quantile(quantile).values.tolist()
#
## 替换异常值为指定的分位数
#    if Q01 > x.min():
#        x=x.copy()
#        x.loc[x<Q01] = Q01
#    if Q99 < x.max():
#        x = x.copy()
#        x.loc[x>Q99] = Q99
#    return(x)