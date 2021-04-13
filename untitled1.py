# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 17:03:17 2020

@author: 92156
"""

import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from sklearn import  preprocessing

cc1=pd.read_excel("数据.xlsx",encoding="gbk")

# 加载 `digits` 数据集

from sklearn.preprocessing import scale
cc1.set_index(['地区'],inplace=True)


# 对`digits.data`数据进行标准化处理
data = scale(cc1)

# print(data)

# 导入 `train_test_split`
from sklearn.model_selection import train_test_split

# 数据分成训练集和测试集
# `test_size`：如果是浮点数，在0-1之间，表示测试子集占比；如果是整数的话就是测试子集的样本数量，`random_state`：是随机数的种子
from sklearn.cluster import KMeans

# 导入“cluster”模块
from sklearn import cluster

# 创建KMeans模型
clf = cluster.KMeans(init='k-means++', n_clusters=3, random_state=42)

# 将训练数据' X_train '拟合到模型中，此处没有用到标签数据y_train，K均值聚类一种无监督学习。
clf.fit(data)
cc1['label'] = clf.labels_  # 对原数据表进行类别标记
c = cc1['label'].value_counts()

print(cc1.values)
cc2=cc1.values

cc2=pd.DataFrame(cc2)


'''
x1	x2	x3	x4	x5	x6	x7	x8	x9	x10	label
5.96	310	461	1557	931	319	44.36	2615	2.2	13631	2
3.39	234	308	1035	498	161	35.02	3052	0.9	12665	1
2.35	157	229	713	295	109	38.4	3031	0.86	9385	1
1.35	81	111	364	150	58	30.45	2699	1.22	7881	0
1.5	88	128	421	144	58	34.3	2808	0.54	7733	0
1.67	86	120	370	153	58	33.53	2215	0.76	7480	0
1.17	63	93	296	117	44	35.22	2528	0.58	8570	0
1.05	67	92	297	115	43	32.89	2835	0.66	7262	0
0.95	64	94	287	102	39	31.54	3008	0.39	7786	0
0.69	39	71	205	61	24	34.5	2988	0.37	11355	0
0.56	40	57	177	61	23	32.62	3149	0.55	7693	0
0.57	58	64	181	57	22	32.95	3202	0.28	6805	0
0.71	42	62	190	66	26	28.13	2657	0.73	7282	0
0.74	42	61	194	61	24	33.06	2618	0.47	6477	0
0.86	42	71	204	66	26	29.94	2363	0.25	7704	0
1.29	47	73	265	114	46	25.93	2060	0.37	5719	0
1.04	53	71	218	63	26	29.01	2099	0.29	7106	0
0.85	53	65	218	76	30	25.63	2555	0.43	5580	0
0.81	43	66	188	61	23	29.82	2313	0.31	5704	0
0.59	35	47	146	46	20	32.83	2488	0.33	5628	0
0.66	36	40	130	44	19	28.55	1974	0.48	9106	0
0.77	43	63	194	67	23	28.81	2515	0.34	4085	0
0.7	33	51	165	47	18	27.34	2344	0.28	7928	0
0.84	43	48	171	65	29	27.65	2032	0.32	5581	0
1.69	26	45	137	75	33	12.1	810	1.0	14199	0
0.55	32	46	130	44	17	28.41	2341	0.3	5714	0
0.6	28	43	129	39	17	31.93	2146	0.24	5139	0
1.39	48	62	208	77	34	22.7	1500	0.42	5377	0
0.64	23	32	93	37	16	28.12	1469	0.34	5415	0
1.48	38	46	151	63	30	17.87	1024	0.38	7368	0
'''

#层次聚类
import sklearn.cluster as sc
model = sc.AgglomerativeClustering(n_clusters=3)
pred_y = model.fit_predict(data)
print(pred_y)
#[2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]



#dbscan

from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
db = DBSCAN(eps=0.3, min_samples=10).fit(data)

from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
# 定义数据集
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# 定义模型
model = DBSCAN(eps=0.003, min_samples=1)
# 模型拟合与聚类预测
yhat = model.fit_predict(data)

labels = db.labels_ 
cc1['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果
cc1.sort_values('cluster_db')

"""
x1	x2	x3	x4	x5	x6	x7	x8	x9	x10	label	cluster_db
5.96	310	461	1557	931	319	44.36	2615	2.2	13631	2	-1
3.39	234	308	1035	498	161	35.02	3052	0.9	12665	1	-1
2.35	157	229	713	295	109	38.4	3031	0.86	9385	1	-1
1.35	81	111	364	150	58	30.45	2699	1.22	7881	0	-1
1.5	88	128	421	144	58	34.3	2808	0.54	7733	0	-1
1.67	86	120	370	153	58	33.53	2215	0.76	7480	0	-1
1.17	63	93	296	117	44	35.22	2528	0.58	8570	0	-1
1.05	67	92	297	115	43	32.89	2835	0.66	7262	0	-1
0.95	64	94	287	102	39	31.54	3008	0.39	7786	0	-1
0.69	39	71	205	61	24	34.5	2988	0.37	11355	0	-1
0.56	40	57	177	61	23	32.62	3149	0.55	7693	0	-1
0.57	58	64	181	57	22	32.95	3202	0.28	6805	0	-1
0.71	42	62	190	66	26	28.13	2657	0.73	7282	0	-1
0.74	42	61	194	61	24	33.06	2618	0.47	6477	0	-1
0.86	42	71	204	66	26	29.94	2363	0.25	7704	0	-1
1.29	47	73	265	114	46	25.93	2060	0.37	5719	0	-1
1.04	53	71	218	63	26	29.01	2099	0.29	7106	0	-1
0.85	53	65	218	76	30	25.63	2555	0.43	5580	0	-1
0.81	43	66	188	61	23	29.82	2313	0.31	5704	0	-1
0.59	35	47	146	46	20	32.83	2488	0.33	5628	0	-1
0.66	36	40	130	44	19	28.55	1974	0.48	9106	0	-1
0.77	43	63	194	67	23	28.81	2515	0.34	4085	0	-1
0.7	33	51	165	47	18	27.34	2344	0.28	7928	0	-1
0.84	43	48	171	65	29	27.65	2032	0.32	5581	0	-1
1.69	26	45	137	75	33	12.1	810	1.0	14199	0	-1
0.55	32	46	130	44	17	28.41	2341	0.3	5714	0	-1
0.6	28	43	129	39	17	31.93	2146	0.24	5139	0	-1
1.39	48	62	208	77	34	22.7	1500	0.42	5377	0	-1
0.64	23	32	93	37	16	28.12	1469	0.34	5415	0	-1
1.48	38	46	151	63	30	17.87	1024	0.38	7368	0	-1
"""

#主成分
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=2)   #降到2维
pca.fit(data)                  #训练
newX=pca.fit_transform(data)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_)  #输出贡献率
print(newX)       
"""
[0.75021586 0.15769872]
[[11.90054102  0.90569015]
 [ 6.04332856 -0.10709543]
 [ 3.55805726 -1.05098759]
 [ 1.02912242 -0.03567673]
 [ 0.84528556 -0.89595961]
 [ 0.81615045  0.01798059]
 [ 0.24670202 -0.58551582]
 [ 0.1252966  -0.86176331]
 [-0.16586276 -1.03372054]
 [-0.31789245 -0.95015871]
 [-0.71138663 -1.31504023]
 [-0.87007458 -1.66369971]
 [-0.75374859 -0.19918966]
 [-0.87722636 -0.86870056]
 [-0.99223783 -0.19120406]
 [-0.83429976  0.51441226]
 [-0.97528034  0.19673745]
 [-1.08551411 -0.14892142]
 [-1.23790448 -0.33533421]
 [-1.37471092 -0.8836958 ]
 [-1.20421678  0.66267649]
 [-1.37864868 -0.65759893]
 [-1.373113    0.10321386]
 [-1.42328713  0.23259868]
 [-0.68278608  4.79514386]
 [-1.66863456 -0.27593556]
 [-1.71062382 -0.47982353]
 [-1.29411821  1.5001984 ]
 [-2.0032759   0.81638157]
 [-1.62964092  2.7949881 ]]
"""