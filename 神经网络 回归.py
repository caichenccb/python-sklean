
import numpy as np
import math
import random
import string
import matplotlib as mpl
import matplotlib.pyplot as plt
 
#random.seed(0)  #当我们设置相同的seed，每次生成的随机数相同。如果不设置seed，则每次会生成不同的随机数
                #参考https://blog.csdn.net/jiangjiang_jian/article/details/79031788
 
#生成区间[a,b]内的随机数
def random_number(a,b):
    return (b-a)*random.random()+a
 
#生成一个矩阵，大小为m*n,并且设置默认零矩阵
def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill]*n)
    return a
 
#函数sigmoid(),这里采用tanh，因为看起来要比标准的sigmoid函数好看
def sigmoid(x):
    return math.tanh(x)
 
#函数sigmoid的派生函数
def derived_sigmoid(x):
    return 1.0 - x**2
 
#构造三层BP网络架构
class BPNN:
    def __init__(self, num_in, num_hidden, num_out):
        #输入层，隐藏层，输出层的节点数
        self.num_in = num_in + 1  #增加一个偏置结点
        self.num_hidden = num_hidden + 1   #增加一个偏置结点
        self.num_out = num_out
        
        #激活神经网络的所有节点（向量）
        self.active_in = [1.0]*self.num_in
        self.active_hidden = [1.0]*self.num_hidden
        self.active_out = [1.0]*self.num_out
        
        #创建权重矩阵
        self.wight_in = makematrix(self.num_in, self.num_hidden)
        self.wight_out = makematrix(self.num_hidden, self.num_out)
        
        #对权值矩阵赋初值
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                self.wight_in[i][j] = random_number(-0.2, 0.2)
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                self.wight_out[i][j] = random_number(-0.2, 0.2)
    
        #最后建立动量因子（矩阵）
        self.ci = makematrix(self.num_in, self.num_hidden)
        self.co = makematrix(self.num_hidden, self.num_out)        
        
    #信号正向传播
    def update(self, inputs):
        if len(inputs) != self.num_in-1:
            raise ValueError('与输入层节点数不符')
            
        #数据输入输入层
        for i in range(self.num_in - 1):
            #self.active_in[i] = sigmoid(inputs[i])  #或者先在输入层进行数据处理
            self.active_in[i] = inputs[i]  #active_in[]是输入数据的矩阵
            
        #数据在隐藏层的处理
        for i in range(self.num_hidden - 1):
            sum = 0.0
            for j in range(self.num_in):
                sum = sum + self.active_in[i] * self.wight_in[j][i]
            self.active_hidden[i] = sigmoid(sum)   #active_hidden[]是处理完输入数据之后存储，作为输出层的输入数据
            
        #数据在输出层的处理
        for i in range(self.num_out):
            sum = 0.0
            for j in range(self.num_hidden):
                sum = sum + self.active_hidden[j]*self.wight_out[j][i]
            self.active_out[i] = sigmoid(sum)   #与上同理
            
        return self.active_out[:]
    
    #误差反向传播
    def errorbackpropagate(self, targets, lr, m):   #lr是学习率， m是动量因子
        if len(targets) != self.num_out:
            raise ValueError('与输出层节点数不符！')
            
        #首先计算输出层的误差
        out_deltas = [0.0]*self.num_out
        for i in range(self.num_out):
            error = targets[i] - self.active_out[i]
            out_deltas[i] = derived_sigmoid(self.active_out[i])*error
        
        #然后计算隐藏层误差
        hidden_deltas = [0.0]*self.num_hidden
        for i in range(self.num_hidden):
            error = 0.0
            for j in range(self.num_out):
                error = error + out_deltas[j]* self.wight_out[i][j]
            hidden_deltas[i] = derived_sigmoid(self.active_hidden[i])*error
        
        #首先更新输出层权值
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                change = out_deltas[j]*self.active_hidden[i]
                self.wight_out[i][j] = self.wight_out[i][j] + lr*change + m*self.co[i][j]
                self.co[i][j] = change
                
        #然后更新输入层权值
        for i in range(self.num_in):
            for i in range(self.num_hidden):
                change = hidden_deltas[j]*self.active_in[i]
                self.wight_in[i][j] = self.wight_in[i][j] + lr*change + m* self.ci[i][j]
                self.ci[i][j] = change
                
        #计算总误差
        error = 0.0
        for i in range(len(targets)):
            error = error + 0.5*(targets[i] - self.active_out[i])**2
        return error
 
    #测试
    def test(self, patterns):
        for i in patterns:
            print(i[0], '->', self.update(i[0]))
    #权重
    def weights(self):
        print("输入层权重")
        for i in range(self.num_in):
            print(self.wight_in[i])
        print("输出层权重")
        for i in range(self.num_hidden):
            print(self.wight_out[i])
            
    def train(self, pattern, itera=100000, lr = 0.1, m=0.1):
        for i in range(itera):
            error = 0.0
            for j in pattern:
                inputs = j[0]
                targets = j[1]
                self.update(inputs)
                error = error + self.errorbackpropagate(targets, lr, m)
            if i % 100 == 0:
                print('误差 %-.5f' % error)
    
#实例
def demo():
    import pandas as pd 
    d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
    y=np.array(d1["chazhio3"])
    x=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","PM10_x","CO_x","NO2_x","SO2_x","O3_x"]])
    patt = [
            [x,y],
      
            ]
    #创建神经网络，3个输入节点，3个隐藏层节点，1个输出层节点
    n = BPNN(11, 3, 1)
    #训练神经网络
    n.train(patt)
    #测试神经网络
    n.test(patt)
    #查阅权重值
    n.weights()
 
     
if __name__ == '__main__':
    demo()
 # 神经网络类
class neuralNetwork():
 
    # 初始化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes   # 输入层节点
        self.hnodes = hiddennodes  # 隐藏层节点
        self.onodes = outputnodes  # 输出层节点
        self.lr = learningrate    # 学习率
        # 三层 神经网络， 权重 数值 (初始值， 随机设置， 【-1， 1】)
        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)  # 输入层和隐藏 的权重矩阵  大小：hnodes * inodes
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)  # 隐层层和输出层 权重矩阵 大小：onodes * hnodes
 
        # # 权重设置 另外方式; 使用 正态概率分布的方式采样；
        # # 1 使用输入层的节点数的开放作为正太分布的标准方差
        # self.wih = (np.random.normal(0.0, pow(self.inodes, -0.5)))
        # self.who = (np.random.normal(0.0, pow(self.hnodes, -0.5)))
        #
        # # 2 使用下一层的节点数的开方作为正太分布的标准方差
        # self.wih = (np.random.normal(0.0, pow(self.hnodes, -0.5)))
        # self.who = (np.random.normal(0.0, pow(self.onodes, -0.5)))
 
        # 定义激活函数
        self.activation_function = lambda x: scipy.special.expit(x)  # 使用了S函数
 
    # 输入数据， 输入数据对应的目标数据
    def train(self, inputs_list, targets_list):
        # 将 其 转化为 二维数组
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
 
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
 
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)  # 一次前馈信号 最终输出结果
 
        # 基于输出 计算误差， 改进权重
        # 误差 是 预期目标输出 - 计算得到的输出
        output_errors = targets - final_outputs
        # 得到 隐藏层的误差
        hidden_errors = np.dot(self.who.T, output_errors)  # 隐藏层和输出层 之间权重矩阵的转置 * 输出误差
        # 更新 隐藏层 和 输出层 之间的权重矩阵
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))
 
        # 更新 输入层 和 隐藏层 之间的权重矩阵
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))
 
        pass
 
    # 查询 神经网络; inputs_list: 输入数据; 即进行预测，返回一次预测结果
    def query(self, inputs_list):
        # 将输入list 转化为 2维数组； T：转置
        inputs = np.array(inputs_list, ndmin=2).T
        # 计算 隐层层的输入，
        hidden_inputs = np.dot(self.wih, inputs)
        # 隐藏层输出信号， 是经过S函数运算后的
        hidden_outputs = self.activation_function(hidden_inputs)
 
        # 计算 输出层的 输入和输出结果
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs  # 返回的使 输出层输出的值
 
    # 封装成 fit(), predict 这种形式
    def fit(self, trains_list, targets_list):
        for index in range(len(trains_list)):
            # 数据 归一化 放在外面
            # scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # 数据归一化 [0.01, 1]
            # 将 标签 对应的 形成 节点数长度的一维数组
            targets = np.zeros(self.onodes) + 0.01  # 期望输入【真实标签值】
            targets[int(targets_list[index])] = 0.99  # 对应标签值 设置为 最大值 0.99； 神经网络 阈值函数 值域在(0， 1)
            self.train(trains_list[index], targets)  # 一行数据训练一次
 
    # 预测函数， 返回预测的标签值
    def predict(self, test_data_list):
        pred_label_list = []
        for raw in test_data_list:
            outputs = self.query(raw)
            label = np.argmax(outputs)
            pred_label_list.append(label)
        pred_label_list = np.asarray(pred_label_list)
        return pred_label_list   
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.special
import pprint
from sklearn import metrics
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
 
 
# 读取小文件 一次读取所有的行
def read_csv_file(filename):
    data_file = open(filename, 'r')
    data_list = data_file.readlines()
    data_file.close()
    return data_list
 
 
def show_digital_image(image_array):
    # cmap 设置 图像色彩； Greys：灰色
    plt.imshow(image_array, cmap="Greys", interpolation="None")
    plt.show()
 
# 符合 Sklearn 库的形式代码
def neural_main1():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    filename = r".\mini_dataset\mnist_train_100.csv"
    all_train_filename = r"./mini_dataset/mnist_train.csv"
    train_data_list = read_csv_file(all_train_filename)
    test_filename = r"./mini_dataset/mnist_test_10.csv"
    all_test_filename = r"./mini_dataset/mnist_test.csv"
    test_data_list = read_csv_file(all_test_filename)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    # 进行 数据归一化； BP神经网络 训练数据和测试数据都要进行数据归一化
    for raw in train_data_list:
        raw = raw.split(",")
        x_train.append((np.asfarray(raw[1:]) / 255.0 * 0.99) + 0.01) # 数据归一化
        y_train.append(int(raw[0]))
 
    for raw in test_data_list:
        raw = raw.split(",")
        x_test.append((np.asfarray(raw[1:]) / 255.0 * 0.99) + 0.01)
        y_test.append(int(raw[0]))
 
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
 
    n.fit(x_train, y_train)
    prd_list = n.predict(x_test)
    accu = metrics.accuracy_score(y_test, prd_list)
    print("预测精度: {}".format(accu))
    return prd_list  # 返回的是预测标签
    # 将数据归一化