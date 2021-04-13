# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:08:10 2019

@author: 92156
"""

from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_excel("工业排污.xls",encoding = 'gbk')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_boxplot(data, start, end):
    fig, ax = plt.subplots(1, end-start, figsize=(24, 4))
    for i in range(start, end):
        sns.boxplot(y=data[data.columns[i]], data=data, ax=ax[i-start])
get_boxplot(df, 1,4 )
def drop_outlier(data, start, end):
    for i in range(start, end):
        field = data.columns[i]
        Q1 = np.quantile(data[field], 0.25)
        Q3 = np.quantile(data[field], 0.75)
        deta = (Q3 - Q1) * 1.5
        data = data[(data[field] >= Q1 - deta) & (data[field] <= Q3 + deta)]
    return data
del_df = drop_outlier(df, 1, 4)
print("原有样本容量:{0}, 剔除后样本容量:{1}".format(df.shape[0], del_df.shape[0]))
get_boxplot(del_df, 1, 4)

km = KMeans(n_clusters=2, random_state=10)
km.fit(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
print(km.cluster_centers_) 
print(km.labels_)
del_df["label"]=km.labels_
del_df["预测"]=km.predict(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
sns.barplot(x=del_df.index, y="label", data=del_df,  palette="Set3")

#看当k取何值时分类较好
import matplotlib.pyplot as plt
K = range(1, 10)
sse = []
for k in K:
    km = KMeans(n_clusters=k, random_state=10)
    km.fit(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
    sse.append(km.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(K, sse, '-o', alpha=0.7)
plt.xlabel("K")
plt.ylabel("SSE")
plt.show()


#当k为5时，看上去簇内离差平方和之和的变化已慢慢变小，那么，我们不妨就将球员聚为7类。如下为聚类效果的代码：
#肘部法则
#计算公式参考
#http://www.360doc.com/content/18/0429/11/47919125_749637065.shtml

km = KMeans(n_clusters=5, random_state=10)
km.fit(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
print(km.cluster_centers_) 
print(km.labels_)
del_df["label"]=km.labels_
del_df["预测"]=km.predict(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
sns.barplot(x=del_df.index, y="label", data=del_df,  palette="Set3")


#轮廓系数
#http://www.360doc.com/content/18/0429/11/47919125_749637065.shtml
#计算公式参看

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn import datasets 
iris=datasets.load_iris
silhouettteScore = []
for i in range(2,15):
    ##构建并训练模型
    kmeans = KMeans(n_clusters = i,random_state=123).fit(iris_data)
    score = silhouette_score(iris_data,kmeans.labels_)
    silhouettteScore.append(score)
plt.figure(figsize=(10,6))
plt.plot(range(2,15),silhouettteScore,linewidth=1.5, linestyle="-")
plt.show()

#层次聚类
import pandas as pd
import seaborn as sns  #用于绘制热图的工具包
from scipy.cluster import hierarchy  #用于进行层次聚类，话层次聚类图的工具包
from scipy import cluster   
import matplotlib.pyplot as plt
from sklearn import decomposition as skldec #用于主成分分析降维的包
Z = hierarchy.linkage(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]], method ='ward',metric='euclidean')
hierarchy.dendrogram(Z,labels = del_df.index)

#利用sns 聚类
sns.clustermap(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]],method ='ward')   
# ’single 最近点算法。   ’complete   这也是最远点算法或Voor Hees算法   ’average   UPGMA算法。 ’weighted  （也称为WPGMA  ’centroid  WPGMC算法。


#聚类评价
#训练聚类模型
from sklearn import datasets 

from sklearn import metrics
iris=datasets.load_iris
model_kmeans=KMeans(n_clusters=3,random_state=0)  #建立模型对象
model_kmeans.fit(x)    #训练聚类模型
y_pre=model_kmeans.predict(x)   #预测聚类模型
 
#评价指标
inertias=model_kmeans.inertia_         #样本距离最近的聚类中心的距离总和
adjusted_rand_s=metrics.adjusted_rand_score(y_true,y_pre)   #调整后的兰德指数
mutual_info_s=metrics.mutual_info_score(y_true,y_pre)       #互信息
adjusted_mutual_info_s=metrics.adjusted_mutual_info_score (y_true,y_pre)  #调整后的互信息
homogeneity_s=metrics.homogeneity_score(y_true,y_pre)   #同质化得分
completeness_s=metrics.completeness_score(y_true,y_pre)   #完整性得分
v_measure_s=metrics.v_measure_score(y_true,y_pre)   #V-measure得分
silhouette_s=metrics.silhouette_score(x,y_pre,metric='euclidean')   #轮廓系数
calinski_harabaz_s=metrics.calinski_harabaz_score(x,y_pre)   #calinski&harabaz得分
print('inertia\tARI\tMI\tAMI\thomo\tcomp\tv_m\tsilh\tc&h')
print('%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d'%
      (inertias,adjusted_rand_s,mutual_info_s,adjusted_mutual_info_s,homogeneity_s,
       completeness_s,v_measure_s,silhouette_s,calinski_harabaz_s))


#inertia    ARI    MI    AMI    homo    comp    v_m    silh    c&h
#300    0.96    1.03    0.94    0.94    0.94    0.94    0.63    2860