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

 
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题
 
sns.set_style('whitegrid',{'font.sans-serif':['Arial Unicode MS','Arial']})
cc2=pd.read_excel("财政收入.xls")
cc2=cc2.drop("年份",axis=1)
sns.pairplot(cc2)
cc2.corr()
model = LinearRegression()
model.fit(cc2[["各项税收X1","经济活动人口X2","国民生产总值X3"]], cc2["财政收入Y"])
display(model.intercept_)  #截距
display(model.coef_)  #线性模型的系数
model = sm.OLS( cc2["财政收入Y"],cc2[["各项税收X1","经济活动人口X2","国民生产总值X3"]])

results = model.fit()
print(results.summary())

#问题二
cc1=pd.read_csv("空置率.csv",encoding="gbk")
sns.pairplot(cc1)
cc1.corr()

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
#排序
cc1=cc1.sort_index(by=["空置率"])
cc1.drop("index",axis=1)

x=cc1["平均租金率"]
y=cc1["空置率"]
a=np.polyfit(x,y,5)#用2次多项式拟合x，y数组
b=np.poly1d(a)#拟合完之后用这个函数来生成多项式对象
c=b(x)#生成多项式对象之后，就是获取x在这个多项式处的值
plt.scatter(x,y,marker='o',label='original datas')#对原始数据画散点图
plt.plot(x,c,ls='--',c='red',label='fitting with second-degree polynomial')#对拟合之后的数据，也就是x，c数组画图
plt.legend()
plt.show()



#问题三
import pandas as pd 
cc3=pd.read_excel("旅游收入数据.xlsx")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split

sns.heatmap(data=cc3)
plt.show()



from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
import numpy as np  
   
from sklearn.datasets import load_iris  
X_train, X_test, y_train, y_test = train_test_split(cc3[["x1","x2","x3","x4","x5","x6"]], cc3["y"], test_size=0.25)

rf=RandomForestRegressor(n_estimators=100,oob_score=True,random_state=42)#这里使用了默认的参数设置  


#随机挑选两个预测不相同的样本  
rf.fit(X_train,y_train)#进行模型的训练  
print(rf.predict(X_test))

print(y_test)


def try_different_method(model):
  model.fit(x_train,y_train)
  score = model.score(x_test, y_test)
  result = model.predict(x_test)
  plt.figure()
  plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
  plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
  plt.title('score: %f'%score)
  plt.legend()
  plt.show()
  
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()
###########4.具体方法调用部分##########
try_different_method(model_RandomForestRegressor)


#岭回归
from sklearn.linear_model import Ridge,RidgeCV
model = RidgeCV(alphas=[0.1, 1.0, 10.0])  # 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
model.fit(X_train, y_train)   # 线性回归建模
# print('系数矩阵:\n',model.coef_)
# print('线性回归模型:\n',model)
# print('交叉验证最佳alpha值',model.alpha_)  # 只有在使用RidgeCV算法时才有效
# 使用模型预测
y_predicted = model.predict(X_test)

# 绘制散点图 参数：x横轴 y纵轴
plt.legend(loc=2)
plt.plot(y_test, y_predicted,c='r')

# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")

# 显示图形
plt.show()
print(model.score(X_train, y_train))
#0.9993310761272257
print(model.score(X_test,y_test))
#0.9319496556561313
asd1as6dasd


'''
saufiasuhfi
asufbiaifujka
asfugiabfuiafjk
'''
