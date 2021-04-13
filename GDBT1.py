import numpy as np  # numpy库
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库
from sklearn.model_selection import train_test_split #导入切分数据
from sklearn.preprocessing import StandardScaler #导入标准化 


# 数据准备
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")  #导入数据集
y=np.array(d1["chazhiso23"])  #选择X值
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]]) #选择Y值
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25)  #切分数据级
ss_X = StandardScaler() #标准化
ss_y = StandardScaler()#标准化
Y_train=np.array(Y_train).reshape(-1,1) #转置  不然会报错 一个warning
Y_test=np.array(Y_test).reshape(-1,1)#转置  不然会报错 一个warning 
X_train = ss_X.fit_transform(X_train)  #模型
X_test = ss_X.transform(X_test)  #模型
Y_train = ss_y.fit_transform(Y_train) #模型
Y_test = ss_y.transform(Y_test) #模型


# 训练回归模型
n_folds = 8  # 设置交叉检验的次数
model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_lr = LinearRegression()  # 建立普通线性回归模型对象
model_etc = ElasticNet()  # 建立弹性网络回归模型对象
model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR']  # 不同模型的名称列表
model_dic = [model_br, model_lr, model_etc, model_svr, model_gbr]  # 不同回归模型对象的集合
cv_score_list = []  # 交叉检验结果列表
pre_y_list = []  # 各个回归模型预测的y值列表
for model in model_dic:  # 读出每个回归模型对象
    scores = cross_val_score(model, X_train, Y_train, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
    cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
    pre_y_list.append(model.fit(X_train, Y_train).predict(X_test))  # 将回归训练中得到的预测y存入列表
    
    
    
# 模型效果指标评估
n_samples, n_features = X.shape  # 总样本量,总特征数
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(5):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(Y_test, pre_y_list[i])  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print ('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
print (70 * '-')  # 打印分隔线
print ('cross validation result:')  # 打印输出标题
print (df1)  # 打印输出交叉检验的数据框
print (70 * '-')  # 打印分隔线
print ('regression metrics:')  # 打印输出标题
print (df2)  # 打印输出回归指标的数据框
print (70 * '-')  # 打印分隔线
print ('short name \t full name')  # 打印输出缩写和全名标题
print ('ev \t explained_variance')
print ('mae \t mean_absolute_error')
print ('mse \t mean_squared_error')
print ('r2 \t r2')
print (70 * '-')  # 打印分隔线


# 模型效果可视化
plt.figure()  # 创建画布
plt.plot(np.arange(X_test.shape[0]),Y_test, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'tan']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(X_test.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像


#取前100的数据
hundred=[]  #取qian yibai 
for i in range(5):
    hundred.append(pre_y_list[i][:100,])
plt.figure()  # 创建画布
plt.plot(np.arange(X_test[:100,].shape[0]),Y_test[:100,], color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'tan']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(hundred):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(X_test[:100,].shape[0]), hundred[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像


#PM2.5
import pandas as pd
asd1=pd.read_excel("PM2.5得分.xlsx")
s=asd1["PM2.5biaozhun"]
ss=asd1["预测PM2.5biaozhun"]
plt.plot(s[:100,],c="r",label='PM2.5')  # 画出每条预测结果线
plt.plot(ss[:100,],c="b",label='PM2.5预测')
plt.title('regression result comparison')  # 标题
plt.legend()
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
