import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
cc1=pa.read_excel["工作簿1.xlsx"]
data=cc1[["第一届","第二届","第三届","第四届"]]
def GM_11_build_model(self, forecast=4):
    if forecast > len(self.data):
        raise Exception('您的数据行不够')
    X_0 = np.array(self.forecast_list['数据'].tail(forecast))
#       1-AGO
    X_1 = np.zeros(X_0.shape)
    for i in range(X_0.shape[0]):
        X_1[i] = np.sum(X_0[0:i+1])
#       紧邻均值生成序列
    Z_1 = np.zeros(X_1.shape[0]-1)
    for i in range(1, X_1.shape[0]):
        Z_1[i-1] = -0.5*(X_1[i]+X_1[i-1])

    B = np.append(np.array(np.mat(Z_1).T), np.ones(Z_1.shape).reshape((Z_1.shape[0], 1)), axis=1)
    Yn = X_0[1:].reshape((X_0[1:].shape[0], 1))

    B = np.mat(B)
    Yn = np.mat(Yn)
    a_ = (B.T*B)**-1 * B.T * Yn

    a, b = np.array(a_.T)[0]

    X_ = np.zeros(X_0.shape[0])
    def f(k):
        return (X_0[0]-b/a)*(1-np.exp(a))*np.exp(-a*(k))

    self.forecast_list.loc[len(self.forecast_list)] = f(X_.shape[0])
def forecast(self, time=5, forecast_data_len=5):
    for i in range(time):
        self.GM_11_build_model(forecast=forecast_data_len)