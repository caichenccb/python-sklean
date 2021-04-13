import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV

d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhio3"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
#y_train = ss_y.fit_transform(y_train)
#y_test = ss_y.transform(y_test)

parameters = {'kernel':('linear', 'rbf',"poly"), 'C':[1, 2, 4,6,9,10,20,40,80,100]}
from sklearn import svm
from sklearn import grid_search
from sklearn.datasets import load_iris
iris = load_iris()
svr = svm.SVR()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X_train, y_train)
print (clf.best_params_)    # 最好的参数