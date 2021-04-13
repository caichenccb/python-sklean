o# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:33:43 2019

@author: 92156
"""

#2019D
#数据描述
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
d1=pd.read_csv("D12.csv",encoding="gbk")
d2=pd.read_csv("D2.csv",encoding="gbk")
cc1=d1.describe()
cc1=d2.describe()
sns.boxplot(y="PM2.5",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="PM10",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="CO",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="NO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="SO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="O3",data=d1,width=0.3,palette="Blues")


#删除异常值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["O3"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
#重复值处理
d1=pd.read_csv("D12.csv",encoding="gbk")
d2=pd.read_csv("D22.csv",encoding="gbk")
d2["shifou"]=d2.duplicated()
d2=d2[d2['shifou'].isin([False])]
d2.to_csv("D23.csv")


#自相关
d2=pd.read_csv("D23.csv",encoding="gbk")
d1=pd.read_csv("D12.csv",encoding="gbk")
import numpy as np
x=np.array(d2["O3"])
def autocorrelation(x,lags):
# 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
# lags表示错开的时间间隔
  n = len(x)
  x = np.array(x)
  result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    /(x[i:].std()*x[:n-i].std()*(n-i)) \
    for i in range(1,lags+1)]
  return result

z = autocorrelation(x, 1000)
print(z)
plt.plot(z)
#零点漂移
import pandas as pd
d1=pd.read_csv("D240dian.csv")
c1=d1.drop_duplicates('xiaoshi')
c1=c1.drop("shijian",axis=1)
c1=c1.drop("xiaoshi",axis=1)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/20000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
#量程漂移
import pandas as pd
d1=pd.read_csv("D240dian.csv")
data =d1.groupby("xiaoshi").mean()

s=[]
A=data.iloc[0:2,5].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/20000)
s=pd.DataFrame(s)
s.to_csv("du.csv")

#匹配
import pandas as pd 
d1=pd.read_excel("tongyihoupiaoyi.xls",encoding="gbk")
data =d1.groupby("xiaoshi").mean()
d2=pd.read_csv("D12.csv",encoding="gbk")
data1=pd.merge(data,d2,on='xiaoshi')
data1=data1.drop("index",1)
data1.to_csv("pipeihoupiaoyi.csv")

#相关性
import pandas as pd
import seaborn as sns 
d1=pd.read_csv("D24shanpiaoyihou.csv",encoding="gbk")
d1=d1.drop("index",axis=1)
cm=d1.corr()
sns.heatmap(cm, vmax=0.001, cmap="Blues")
#
import pandas as pd
d1=pd.read_csv("D26chazhiso2.csv",encoding="gbk")
import seaborn as sns 
sns.pairplot(d1)

#打分
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
d1=pd.read_excel("PM2.5yuPM10.xlsx",encoding="gbk")
Q=d1["PM2.5_y"]
B=d1["预测PM2.5"]
Y=d1["PM10_y"]
Z=d1["预测PM10"]
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(5):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(Y, Z)  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
df2 = pd.DataFrame(model_metrics_list, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print (70 * '-')  # 打印分隔线
print ('regression metrics:')  # 打印输出标题
print (df2)  # 打印输出回归指标的数据框

#%画图 PM2.5 /15  00：14
plt.plot(rbf_svr_y_predict[:100,],c="r")
plt.plot(y_test[:100,],c="b")
plt.xlabel("测试集")
plt.ylabel("标准化处理后的数据") 
plt.title("PM2.5数据预测")