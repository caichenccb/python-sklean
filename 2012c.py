# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 09:38:45 2019

@author: 92156
"""
#数据清洗
import time, datetime
import pandas as pd
from dateutil import parser
tss1=pd.read_excel("data1(8.25.14).xls",dtype = 'str')
tss2=pd.read_excel("data2.xls",dtype = 'str')
tss3=pd.read_excel("data3.xls",dtype = 'str')
tss4=pd.read_excel("data4.xls",dtype = 'str')
tss1=tss1.dropna(subset=["Report time"])
tss1=tss1.dropna(subset=["Time of incidence"])
cc1=[]
cc2=[]
for i in tss1['Report time']:
    cc1.append(str(parser.parse(i)))
tss1['Report time']=cc1
for i in tss1['Time of incidence']:
    cc2.append(str(parser.parse(i)))
tss1['Time of incidence']=cc2
tss1.to_excel('data1(8.25.14).xls')

tss2=tss2.dropna(subset=['Report time'])
tss2=tss2.dropna(subset=['Time of incidence'])

cc3=[]
cc4=[]
for i in tss2['Report time']:
    cc3.append(str(parser.parse(i)))
tss2['Report time']=cc3
for i in tss2['Time of incidence']:
    cc4.append(str(parser.parse(i)))
tss2['Time of incidence']=cc4
tss2.to_excel('data2(8.25.14).xls')


tss3=tss3.dropna(subset=['Report time'])
tss3=tss3.dropna(subset=['Time of incidence'])

cc5=[]
cc6=[]
for i in tss3['Report time']:
    cc5.append(str(parser.parse(i)))
tss3['Report time']=cc5
for i in tss3['Time of incidence']:
    cc6.append(str(parser.parse(i)))
tss3['Time of incidence']=cc6
tss3.to_excel('data3(8.25.14).xls')

tss4=tss4.dropna(subset=['Report time'])
tss4=tss4.dropna(subset=['Time of incidence'])

cc7=[]
cc8=[]
for i in tss4['Report time']:
    cc7.append(str(parser.parse(i)))
tss4['Report time']=cc7
for i in tss4['Time of incidence']:
    cc8.append(str(parser.parse(i)))
tss4['Time of incidence']=cc8
tss4.to_excel('data4(8.25.14).xls')


#检验正太分布
import scipy.stats as stats
cc1=pd.read_excel("data51cqb.xlsx")
cc1.index=cc1["xx"]
xx=[]
del cc1['xx']
for i in range(8):
    xx.append(stats.shapiro(cc1.iloc[i]))
#(0.9402639865875244, 0.5014996528625488)
#(0.9289568662643433, 0.3691868782043457)
#(0.9468663930892944, 0.5917196273803711)
#(0.9384476542472839, 0.47822731733322144)
#(0.9433080554008484, 0.5420776009559631)
#(0.9273831248283386, 0.3531990349292755)
#(0.9095460176467896, 0.21050703525543213)
#(0.949221134185791, 0.6256268620491028)    
    
#相关系数检验
#皮尔逊相关性系数 
#http://blog.sina.com.cn/s/blog_69e75efd0102wmd2.html
    
    
#    https://blog.csdn.net/qq_25174673/article/details/89606487
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
cc1=pd.read_excel("data5rzsyijidu.xlsx")
cc1.index=cc1["riqi"]

del cc1['riqi']
corrs=cc1
cm=corrs.corr().round(2)
sns.set(font_scale=0.8) 

print(cm)  #font_scale设置字体大小
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象
sns.heatmap(cm,annot = True,cmap = 'Blues')

plt.tight_layout()

plt.figure(figsize=(20, 6.5))
#plt.savefig('corr_mat.png', dpi=300)
plt.show()

#回归模型


#不太好
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
data = pd.read_excel("data5rzsyijidu.xlsx")
y = data['lisanchengdu']
x1 = data['Aver pres']
x2 = data['High pres']
x3 = data['Low pres']
x4 = data['Aver temp']
x5 = data['High temp']
x6 = data['Low temp']
x7 = data['Aver RH']
x8 = data['Min RH']

x = np.column_stack((x1, x2,x3,x4,x5,x6,x7,x8))#表示自变量包含x1和x1的平方，如果需要更高次，可再做添加，也可以添加交叉项x1*y，或者其他变量
#注意自变量设置时，每项高次都需依次设置,x1**2表示乘方，或者x1*x1也可以
x = sm.add_constant(x) #增一列1，用于分析截距
model = sm.OLS(y, x)
#数据拟合，生成模型
result = model.fit()
print(result.summary())
#应用模型预测
Y=result.fittedvalues

#不行
#基于最小二乘法
from sklearn.linear_model import LinearRegression
import pandas as pd
data = pd.read_excel("data5rzsyijidu.xlsx")
lr = LinearRegression()
X=data[["Aver pres","High pres","Low pres","Aver temp","High temp","Low temp","Aver RH","Min RH"]]
y = data['lisanchengdu']
lr.fit(X, y)#拟合模型
lr_y_predict = lr.predict(X)#做预测

#buxing
#https://blog.csdn.net/dongyanwen6036/article/details/78300755
#广义线性模型
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import datasets, linear_model
data = pd.read_excel("data5rzsyijidu.xlsx")
X=data[["Aver pres","High pres","Low pres","Aver temp","High temp","Low temp","Aver RH","Min RH"]]
y = data['lisanchengdu']

lr = linear_model.LinearRegression()
lr.fit(X,y)
print('Coefficients:%s, intercept %.2f'%(lr.coef_,lr.intercept_))

#弹性网
from sklearn.linear_model import ElasticNetCV,ElasticNet



#2.岭回归
#不行
from sklearn.linear_model import Ridge
import pandas as pd
data = pd.read_excel("data5rzsyijidu.xlsx")
lr = Ridge()
X=data[["Aver pres","High pres","Low pres","Aver temp","High temp","Low temp","Aver RH","Min RH"]]
y = data['lisanchengdu']
lr.fit(X, y)#拟合模型
lr_y_predict = lr.predict(X)#做预测


#PAC 降维
import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split 
data = pd.read_excel("工作簿1(2).xlsx")
X=data[["Aver pres","High pres","Low pres","Aver temp","High temp","Low temp","Aver RH","Min RH"]]
Y = data['患者人数离散程度']
# 对数据进行标准化
X = (X - X.mean())/np.std(X)
Y = (Y - Y.mean())/np.std(Y)
# 创建pca模型
pca = PCA(n_components=1)
# 对模型进行训练
pca.fit(X)
# 返回降维后据
X = pca.transform(X)

# 使用返回后的数据用线性回归模型进行建模
import statsmodels.api as sm
ols = sm.OLS(Y, X).fit()
ols.summary()


from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
del_df=pd.read_csv("rizhenduan.csv",encoding = 'gbk')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

km = KMeans(n_clusters=4, random_state=10)
km.fit(del_df[["Rizhenhengduanliang"]])
print(km.cluster_centers_) 
print(km.labels_)
del_df["label"]=km.labels_
del_df["预测"]=km.predict(del_df[["Rizhenhengduanliang"]])
sns.barplot(x=del_df.index, y="label", data=del_df,  palette="Set3")

#看当k取何值时分类较好
import matplotlib.pyplot as plt
K = range(1, 10)
sse = []
for k in K:
    km = KMeans(n_clusters=k, random_state=10)
    km.fit(del_df[["Rizhenhengduanliang"]])
    sse.append(km.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(K, sse, '-o', alpha=0.7)
plt.xlabel("K")
plt.ylabel("SSE")
plt.show()


#当k为5时，看上去簇内离差平方和之和的变化已慢慢变小，那么，我们不妨就将球员聚为7类。如下为聚类效果的代码：

km = KMeans(n_clusters=5, random_state=10)
km.fit(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
print(km.cluster_centers_) 
print(km.labels_)
del_df["label"]=km.labels_
del_df["预测"]=km.predict(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
sns.barplot(x=del_df.index, y="label", data=del_df,  palette="Set3")






