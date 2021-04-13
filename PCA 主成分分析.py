# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:09:11 2019

@author: 92156
"""

import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X=pd.read_excel("消费结构.xls")
pp=X.describe()
pca = PCA(n_components=0.9)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
X_new = pca.fit_transform(X)
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(X_new[:,0],X_new[:,1],X_new[:,2])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

#计算方差贡献率，和综合因素
import pandas as pd 
import numpy as np
cc1=pd.read_excel("cc1.xlsx")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(cc1)
scaler
X_scaler=pd.DataFrame(scaler.transform(cc1))
pca = PCA(n_components=0.9) #n_components提取因子数量

pca.fit(X_scaler)

pca.explained_variance_

pca.explained_variance_ratio_
pca.components_
k1_spss=pca.components_/np.sqrt(pca.explained_variance_.reshape(2,1))
x_tf=pca.transform(X_scaler)
scaler2=StandardScaler().fit(x_tf)
scaler2=StandardScaler().fit(x_tf)
x_tf_scaler=pd.DataFrame(scaler2.transform(x_tf))
k_sign=np.sign(k1_spss.sum(axis=1))

x_tf_scaler_sign=x_tf_scaler*k_sign #取正负号

rat=pca.explained_variance_ratio_

x_tf_scaler_sign['FAC_score']=np.sum(x_tf_scaler_sign*rat,axis=1)

x_tf_scaler_sign
import random
resultlist=random.sample(range(0, 7),3)
X_new=pd.DataFrame(X_new)
X_new.index=X.columns
ccc=X.iloc[resultlist]*X_new[0]
ccc1=X.iloc[resultlist]*X_new[1]
ccc2=X.iloc[resultlist]*X_new[2]
ccc.loc['Row_sum'] = ccc.apply(lambda x: x.sum())   #列和
ccc['Col_sum'] = ccc.apply(lambda x: x.sum(), axis=1)   #行和
ccc1.loc['Row_sum'] = ccc1.apply(lambda x: x.sum())   #列和
ccc1['Col_sum'] = ccc1.apply(lambda x: x.sum(), axis=1)   #行和
ccc2.loc['Row_sum'] = ccc2.apply(lambda x: x.sum())   #列和
ccc2['Col_sum'] = ccc2.apply(lambda x: x.sum(), axis=1)   #行和
