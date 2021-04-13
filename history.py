# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Thu Sep  5 21:26:47 2019)---
import pandas as pd 
cc1=pd.read_excel("cc1.xlsx")
import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X=cc1
pca = PCA(n_components=0.9)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
X_new = pca.fit_transform(X)
fig=plt.figure()
pca = PCA(n_components=4)
pca.fit(X)
print(pca.explained_variance_ratio_)
pca.n_components
pca.score
%reset
import pandas as pd 
cc1=pd.read_excel("cc1.xlxs")
cc1=pd.read_excel("cc1.xlsx")
rom __future__ import print_function

import math


# atan函数转换方法
data4 = [math.atan(x) for x in data]
from __future__ import print_function

import math


# atan函数转换方法
cc1 = [math.atan(x) for x in cc1]

cc1
cc1=pd.read_excel("cc1.xlxs")
cc1=pd.read_excel("cc1.xlsx")
from __future__ import print_function

import math


# atan函数转换方法
cc1 = [math.atan(x) for x in cc1]

import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3)
X=cc1
pca.fit(X)
X=np.array(X)
pca.fit(X)
X.reshape(-1,1)
pca.fit(X)
X=X.reshape(-1,1)
pca.fit(X)
pca = PCA(n_components=0.9)
pca.fit(X)
X_new = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
%reset
import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cc1=pd.read_excel("cc1.xlsx")
from __future__ import print_function

import math


# atan函数转换方法
cc1 = [math.atan(x) for x in cc1]

X=np.array(cc1).reshape(-1,1)
pca = PCA(n_components=0.9)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
X_new = pca.fit_transform(X)
import pandas as pd
import pylab as pl
from sklearn import datasets
from sklearn.decomposition import PCA

# load dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# normalize data
df_norm = (df - df.mean()) / df.std()

# PCA
pca = PCA(n_components=2)
pca.fit_transform(df_norm.values)
print (pca.explained_variance_ratio_)

pca.components_
pca.fit(X)
%reset

## ---(Fri Sep  6 13:53:09 2019)---
import pandas as pd 
cc1=pd.read_excel("cc1.xlsx")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(cc1)
scaler
pca = PCA(n_components=0.9)
pca.fit(scaler)
X_scaler=pd.DataFrame(scaler.transform(df))
X_scaler=pd.DataFrame(scaler.transform(cc1))
pca = PCA(n_components=0.9) #n_components提取因子数量

pca.fit(X_scaler)

pca.explained_variance_

pca.explained_variance_ratio_
pca.components_
k1_spss=pca.components_/np.sqrt(pca.explained_variance_.reshape(2,1))
import numpy as np
k1_spss=pca.components_/np.sqrt(pca.explained_variance_.reshape(2,1))
x_tf=pca.transform(X_scaler)
scaler2=preprocessing.StandardScaler().fit(x_tf)
scaler2=StandardScaler().fit(x_tf)
x_tf_scaler=pd.DataFrame(scaler2.transform(x_tf))
k_sign=np.sign(k1_spss.sum(axis=1))

x_tf_scaler_sign=x_tf_scaler*k_sign #取正负号

rat=pca.explained_variance_ratio_

x_tf_scaler_sign['FAC_score']=np.sum(x_tf_scaler_sign*rat,axis=1)

x_tf_scaler_sign
%reset
import pandas as pd 
cc1=pd.read_excel("cc1.xlsx")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(cc1)
scaler
pca = PCA(n_components=0.9)
pca.fit(scaler)
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
import pandas as pd 
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

## ---(Fri Sep  6 20:31:43 2019)---

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn import cross_validation

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

##变为2分类
X, y = X[y != 2], y[y != 2]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.3,random_state=0)

# Learn to predict each class against the other
svm = svm.SVC(kernel='linear', probability=True,random_state=random_state)

###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = svm.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
         
runfile('C:/Users/92156/.spyder-py3/untitled0.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/KFold，StratifiedKFold k折交叉切分.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/AUC.py', wdir='C:/Users/92156/.spyder-py3')

## ---(Sat Sep  7 20:55:23 2019)---
import geatpy as ea
import numpy
numpy.__version__
import numpy
import geatpy as ea
import Geatpy
import geatpy
import geatpy as ga
help(ga.ranking)
# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

#Importing required modules
import math
import random
import matplotlib.pyplot as plt

#First function to optimize
def function1(x):
    value = -x**2
    return value

#Second function to optimize
def function2(x):
    value = -(x-2)**2
    return value

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

#Function to carry out the crossover
def crossover(a,b):
    r=random.random()
    if r>0.5:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2)

#Function to carry out the mutation operator
def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob <1:
        solution = min_x+(max_x-min_x)*random.random()
    return solution

#Main program starts here
pop_size = 20
max_gen = 921

#Initialization
min_x=-55
max_x=55
solution=[min_x+(max_x-min_x)*random.random() for i in range(0,pop_size)]
gen_no=0
while(gen_no<max_gen):
    function1_values = [function1(solution[i])for i in range(0,pop_size)]
    function2_values = [function2(solution[i])for i in range(0,pop_size)]
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    print("The best front for Generation number ",gen_no, " is")
    for valuez in non_dominated_sorted_solution[0]:
        print(round(solution[valuez],3),end=" ")
    print("\n")
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
    solution2 = solution[:]
    #Generating offsprings
    while(len(solution2)!=2*pop_size):
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)
        solution2.append(crossover(solution[a1],solution[b1]))
    function1_values2 = [function1(solution2[i])for i in range(0,2*pop_size)]
    function2_values2 = [function2(solution2[i])for i in range(0,2*pop_size)]
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1

#Lets plot the final front now
function1 = [i * -1 for i in function1_values]
function2 = [j * -1 for j in function2_values]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()






import numpy as np
import geatpy as ea

class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self):
        name = 'ZDT1' # 初始化name（函数名称，可以随意设置）
        M = 2 # 初始化M（目标维数）
        maxormins = [1] * M # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 30 # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim # 决策变量下界
        ub = [1] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界
        ubin = [1] * Dim # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop): # 目标函数
        Vars = pop.Phen # 得到决策变量矩阵
        ObjV1 = Vars[:, 0]
        gx = 1 + 9 * np.sum(Vars[:, 1:30], 1)
        hx = 1 - np.sqrt(ObjV1 / gx)
        ObjV2 = gx * hx
        pop.ObjV = np.array([ObjV1, ObjV2]).T # 把结果赋值给ObjV

    def calBest(self): # 计算全局最优解
        N = 10000 # 生成10000个参考点
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        globalBestObjV = np.array([ObjV1, ObjV2]).T

        return globalBestObjV
    
import geatpy as ea # import geatpy


"""================================实例化问题对象============================="""
problem = MyProblem()     # 生成问题对象
"""==================================种群设置================================"""
Encoding = 'RI'           # 编码方式
NIND = 50                 # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
"""================================算法参数设置==============================="""
myAlgorithm = ea.moea_NSGA2_templet(problem, population) # 实例化一个算法模板对象`
myAlgorithm.MAXGEN = 200  # 最大进化代数
myAlgorithm.drawing = 1   # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制过程动画）
"""===========================调用算法模板进行种群进化===========================
调用run执行算法模板，得到帕累托最优解集NDSet。NDSet是一个种群类Population的对象。
NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
详见Population.py中关于种群类的定义。
"""
NDSet = myAlgorithm.run() # 执行算法模板，得到非支配种群
NDSet.save()              # 把结果保存到文件中
# 输出
print('用时：%f 秒'%(myAlgorithm.passTime))
print('评价次数：%d 次'%(myAlgorithm.evalsNum))
print('非支配个体数：%d 个'%(NDSet.sizes))
print('单位时间找到帕累托前沿点个数：%d 个'%(int(NDSet.sizes // myAlgorithm.passTime)))
# 计算指标
PF = problem.getBest() # 获取真实前沿，详见Problem.py中关于Problem类的定义
if PF is not None and NDSet.sizes != 0:
    GD = ea.indicator.GD(NDSet.ObjV, PF)       # 计算GD指标
    IGD = ea.indicator.IGD(NDSet.ObjV, PF)     # 计算IGD指标
    HV = ea.indicator.HV(NDSet.ObjV, PF)       # 计算HV指标
    Spacing = ea.indicator.Spacing(NDSet.ObjV) # 计算Spacing指标
    print('GD',GD)
    print('IGD',IGD)
    print('HV', HV)
    print('Spacing', Spacing)
"""=============================进化过程指标追踪分析============================"""
if PF is not None:
    metricName = [['IGD'], ['HV']]
    [NDSet_trace, Metrics] = ea.indicator.moea_tracking(myAlgorithm.pop_trace, PF, metricName, problem.maxormins)
    # 绘制指标追踪分析图
    ea.trcplot(Metrics, labels = metricName, titles = metricName)
    
creator.create('MultiObjMin', base.Fitness, weights=(-1.0, -1.0))
creator.create('Individual', list, fitness=creator.MultiObjMin)

## 个体编码
def uniform(low, up):
    # 用均匀分布生成个体
    return [random.uniform(a,b) for a,b in zip(low, up)]

toolbox = base.Toolbox()
NDim = 2 # 变量数为2
low = [0]*NDim # 变量下界
up = [1]*NDim # 变量上界

toolbox.register('attr_float', uniform, low, up)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

## 评价函数
def ZDT3(ind):
    # ZDT3评价函数，ind长度为2
    n = len(ind)
    f1 = ind[0]
    g = 1 + 9 * np.sum(ind[1:]) / (n-1)
    f2 = g * (1 - np.sqrt(ind[0]/g) - ind[0]/g * np.sin(10*np.pi*ind[0]))
    return f1, f2

toolbox.register('evaluate', ZDT3)

## 注册工具
toolbox.register('selectGen1', tools.selTournament, tournsize=2)
toolbox.register('select', tools.emo.selTournamentDCD) # 该函数是binary tournament，不需要tournsize
toolbox.register('mate', tools.cxSimulatedBinaryBounded, eta=20.0, low=low, up=up)
toolbox.register('mutate', tools.mutPolynomialBounded, eta=20.0, low=low, up=up, indpb=1.0/NDim)

## 遗传算法主程序
# 参数设置
toolbox.popSize = 100
toolbox.maxGen = 200
toolbox.cxProb = 0.7
toolbox.mutateProb = 0.2

# 迭代部分
# 第一代
pop = toolbox.population(toolbox.popSize) # 父代
fitnesses = toolbox.map(toolbox.evaluate, pop)
for ind, fit in zip(pop,fitnesses):
    ind.fitness.values = fit
fronts = tools.emo.sortNondominated(pop, k=toolbox.popSize)
# 将每个个体的适应度设置为pareto前沿的次序
for idx, front in enumerate(fronts):
    for ind in front:
        ind.fitness.values = (idx+1),
# 创建子代
offspring = toolbox.selectGen1(pop, toolbox.popSize) # binary Tournament选择
offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxProb, toolbox.mutateProb)

# 第二代之后的迭代
for gen in range(1, toolbox.maxGen):
    combinedPop = pop + offspring # 合并父代与子代
    # 评价族群
    fitnesses = toolbox.map(toolbox.evaluate, combinedPop)
    for ind, fit in zip(combinedPop,fitnesses):
        ind.fitness.values = fit
    # 快速非支配排序
    fronts = tools.emo.sortNondominated(combinedPop, k=toolbox.popSize, first_front_only=False)
    # 拥挤距离计算
    for front in fronts:
        tools.emo.assignCrowdingDist(front)
    # 环境选择 -- 精英保留
    pop = []
    for front in fronts:
        pop += front
    pop = toolbox.clone(pop)
    pop = tools.selNSGA2(pop, k=toolbox.popSize, nd='standard')
    # 创建子代
    offspring = toolbox.select(pop, toolbox.popSize)
    offspring = toolbox.clone(offspring)
    offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxProb, toolbox.mutateProb
    
    
creator.create('MultiObjMin', base.Fitness, weights=(-1.0, -1.0))
creator.create('Individual', list, fitness=creator.MultiObjMin)

## 个体编码
def uniform(low, up):
    # 用均匀分布生成个体
    return [random.uniform(a,b) for a,b in zip(low, up)]

toolbox = base.Toolbox()
NDim = 2 # 变量数为2
low = [0]*NDim # 变量下界
up = [1]*NDim # 变量上界

toolbox.register('attr_float', uniform, low, up)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

## 评价函数
def ZDT3(ind):
    # ZDT3评价函数，ind长度为2
    n = len(ind)
    f1 = ind[0]
    g = 1 + 9 * np.sum(ind[1:]) / (n-1)
    f2 = g * (1 - np.sqrt(ind[0]/g) - ind[0]/g * np.sin(10*np.pi*ind[0]))
    return f1, f2

toolbox.register('evaluate', ZDT3)

## 注册工具
toolbox.register('selectGen1', tools.selTournament, tournsize=2)
toolbox.register('select', tools.emo.selTournamentDCD) # 该函数是binary tournament，不需要tournsize
toolbox.register('mate', tools.cxSimulatedBinaryBounded, eta=20.0, low=low, up=up)
toolbox.register('mutate', tools.mutPolynomialBounded, eta=20.0, low=low, up=up, indpb=1.0/NDim)

## 遗传算法主程序
# 参数设置
toolbox.popSize = 100
toolbox.maxGen = 200
toolbox.cxProb = 0.7
toolbox.mutateProb = 0.2

# 迭代部分
# 第一代
pop = toolbox.population(toolbox.popSize) # 父代
fitnesses = toolbox.map(toolbox.evaluate, pop)
for ind, fit in zip(pop,fitnesses):
    ind.fitness.values = fit
fronts = tools.emo.sortNondominated(pop, k=toolbox.popSize)
# 将每个个体的适应度设置为pareto前沿的次序
for idx, front in enumerate(fronts):
    for ind in front:
        ind.fitness.values = (idx+1),
# 创建子代
offspring = toolbox.selectGen1(pop, toolbox.popSize) # binary Tournament选择
offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxProb, toolbox.mutateProb)

# 第二代之后的迭代
for gen in range(1, toolbox.maxGen):
    combinedPop = pop + offspring # 合并父代与子代
    # 评价族群
    fitnesses = toolbox.map(toolbox.evaluate, combinedPop)
    for ind, fit in zip(combinedPop,fitnesses):
        ind.fitness.values = fit
    # 快速非支配排序
    fronts = tools.emo.sortNondominated(combinedPop, k=toolbox.popSize, first_front_only=False)
    # 拥挤距离计算
    for front in fronts:
        tools.emo.assignCrowdingDist(front)
    # 环境选择 -- 精英保留
    pop = []
    for front in fronts:
        pop += front
    pop = toolbox.clone(pop)
    pop = tools.selNSGA2(pop, k=toolbox.popSize, nd='standard')
    # 创建子代
    offspring = toolbox.select(pop, toolbox.popSize)
    offspring = toolbox.clone(offspring)
    offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxProb, toolbox.mutateProb)

# front = tools.emo.sortNondominated(pop, len(pop))[0]
for ind in front:
    plt.plot(ind.fitness.values[0], ind.fitness.values[1], 'r.', ms=2)
for ind in gridPop:
    plt.plot(ind.fitness.values[0], ind.fitness.values[1], 'b.', ms=1)    
plt.title('Pareto optimal front derived with NSGA-II', fontsize=12)
plt.xlabel('$f_1$')
plt.ylabel('$f_2$')
plt.tight_layout()
plt.savefig('Pareto_optimal_front_derived_with_NSGA-II.png')
creator
from deap import base, creator, tools
from scipy.stats import bernoulli

# 定义问题
creator.create('FitnessMin', base.Fitness, weights=(-1.0,)) # 单目标，最小化
creator.create('Individual', list, fitness = creator.FitnessMin)

# 生成个体
GENE_LENGTH = 5
toolbox = base.Toolbox() #实例化一个Toolbox
toolbox.register('Binary', bernoulli.rvs, 0.5)
toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Binary, n=GENE_LENGTH)

# 生成初始族群
N_POP = 10
toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)
toolbox.Population(n = N_POP)

from deap import base, creator, tools
import numpy as np
# 定义问题
creator.create('FitnessMin', base.Fitness, weights=(-1.0,)) #优化目标：单变量，求最小值
creator.create('Individual', list, fitness = creator.FitnessMin) #创建Individual类，继承list

# 生成个体
IND_SIZE = 5
toolbox = base.Toolbox()
toolbox.register('Attr_float', np.random.rand)
toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Attr_float, n=IND_SIZE)

# 生成初始族群
N_POP = 10
toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)
pop = toolbox.Population(n = N_POP)

# 定义评价函数
def evaluate(individual):
  return sum(individual), #注意这个逗号，即使是单变量优化问题，也需要返回tuple

# 评价初始族群
toolbox.register('Evaluate', evaluate)
fitnesses = map(toolbox.Evaluate, pop)
for ind, fit in zip(pop, fitnesses):
  ind.fitness.values = fit

# 选择方式1：锦标赛选择
toolbox.register('TourSel', tools.selTournament, tournsize = 2) # 注册Tournsize为2的锦标赛选择
selectedTour = toolbox.TourSel(pop, 5) # 选择5个个体
print('锦标赛选择结果：')
for ind in selectedTour:
  print(ind)
  print(ind.fitness.values)

# 选择方式2: 轮盘赌选择
toolbox.register('RoulSel', tools.selRoulette)
selectedRoul = toolbox.RoulSel(pop, 5)
print('轮盘赌选择结果：')
for ind in selectedRoul:
  print(ind)
  print(ind.fitness.values)

# 选择方式3: 随机普遍抽样选择
toolbox.register('StoSel', tools.selStochasticUniversalSampling)
selectedSto = toolbox.StoSel(pop, 5)
print('随机普遍抽样选择结果：')
for ind in selectedSto:
  print(ind)
  print(ind.fitness.values)
  
def genCity(n, Lb = 100 ,Ub = 999):
    # 生成城市坐标
    # 输入：n -- 需要生成的城市数量
    # 输出: nx2 np array 每行是一个城市的[X,Y]坐标
    np.random.seed(42) # 保证结果的可复现性
    return np.random.randint(low = Lb, high = Ub, size=(n,2))

import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
from scipy.spatial import distance
from deap import creator, base, tools, algorithms
import random

params = {
    'font.family': 'serif',
    'figure.figsize': [4.0, 3.0],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'legend.fontsize': 'small'
}
plt.rcParams.update(params)

genCity(n=100)
cc=genCity(n=100)
runfile('C:/Users/92156/.spyder-py3/遗传算法 TSP.py', wdir='C:/Users/92156/.spyder-py3')
tour,tourDist = nnTSP(cities)
print('nnTSP寻找到的最优路径为：' + str(tour))
print('nnTSP寻找到最优路径长度为：' + str(tourDist))
plotTour(tour, cities)

%reset
runfile('C:/Users/92156/.spyder-py3/遗传算法 TSP.py', wdir='C:/Users/92156/.spyder-py3')
%reset
def nearestNeighbor(cityDist, currentPos, unvisited):
    # 输入：cityDist -- [n,n] np array,记录城市间距离
    # currentPos -- 一个数字，指示当前位于的城市
    # unvisited -- 一个列表，指示可选邻居列表
    # 输出：
    # nextToVisit -- 一个数字，给出最近邻的index
    # dist -- 一个数字，给出到最近邻的距离
    neighborDist = cityDist[currentPos,unvisited]
    neighborIdx = np.argmin(neighborDist)
    nextToVisit = unvisited[neighborIdx]
    dist = neighborDist[neighborIdx]
    return nextToVisit, dist

# 用贪婪算法求解TSP
def nnTSP(cities, start = 0):
    cityList = list(range(cities.shape[0]))
    tour = [start]
    unvisited = cityList.copy()
    unvisited.remove(start)
    currentPos = start
    tourDist = 0
    cityDist = cityDistance(cities)
    while unvisited:
        # 当unvisited集合不为空时，重复循环
        # 找到当前位置在unvisited中的最近邻
        nextToVisit, dist = nearestNeighbor(cityDist, currentPos, unvisited)
        tourDist += dist
        currentPos = nextToVisit
        tour.append(nextToVisit)
        unvisited.remove(nextToVisit)
    # 重新回到起点
    tour.append(start)
    tourDist += cityDist[currentPos, start]
    return tour,tourDist

tour,tourDist = nnTSP(cities)
print('nnTSP寻找到的最优路径为：' + str(tour))
print('nnTSP寻找到最优路径长度为：' + str(tourDist))
plotTour(tour, cities)

cities = genCity(nCities) # 随机生成nCities个城市坐标
def genCity(n, Lb = 100 ,Ub = 999):
    # 生成城市坐标
    # 输入：n -- 需要生成的城市数量
    # 输出: nx2 np array 每行是一个城市的[X,Y]坐标
    np.random.seed(42) # 保证结果的可复现性
    return np.random.randint(low = Lb, high = Ub, size=(n,2))


# 计算并存储城市距离矩阵
def cityDistance(cities):
    # 生成城市距离矩阵 distMat[A,B] = distMat[B,A]表示城市A，B之间距离
    # 输入：cities -- [n,2] np array， 表示城市坐标
    # 输出：nxn np array， 存储城市两两之间的距离
    return distance.cdist(cities, cities, 'euclidean')


def completeRoute(individual):
    # 序列编码时，缺少最后一段回到原点的线段
    return individual + [individual[0]] # 不要用append


# 计算给定路线的长度
def routeDistance(route):
    # 输入：
    #      route -- 一条路线，一个sequence
    # 输出：routeDist -- scalar，路线的长度
    if route[0] != route[-1]:
        route = completeRoute(route)
    routeDist = 0
    for i,j in zip(route[0::],route[1::]):
        routeDist += cityDist[i,j] # 这里直接从cityDist变量中取值了，其实并不是很安全的写法，单纯偷懒了
    return (routeDist), # 注意DEAP要求评价函数返回一个元组


# 路径可视化
def plotTour(tour, cities, style = 'bo-'):
    if len(tour)>1000: plt.figure(figsize = (15,10))
    start = tour[0:1]
    for i,j in zip(tour[0::], tour[1::]):
        plt.plot([cities[i,0],cities[j,0]], [cities[i,1],cities[j,1]], style)
    plt.plot(cities[start,0],cities[start,1],'rD')
    plt.axis('scaled')
    plt.axis('off')


def nearestNeighbor(cityDist, currentPos, unvisited):
    # 输入：cityDist -- [n,n] np array,记录城市间距离
    # currentPos -- 一个数字，指示当前位于的城市
    # unvisited -- 一个列表，指示可选邻居列表
    # 输出：
    # nextToVisit -- 一个数字，给出最近邻的index
    # dist -- 一个数字，给出到最近邻的距离
    neighborDist = cityDist[currentPos,unvisited]
    neighborIdx = np.argmin(neighborDist)
    nextToVisit = unvisited[neighborIdx]
    dist = neighborDist[neighborIdx]
    return nextToVisit, dist


# 用贪婪算法求解TSP
def nnTSP(cities, start = 0):
    cityList = list(range(cities.shape[0]))
    tour = [start]
    unvisited = cityList.copy()
    unvisited.remove(start)
    currentPos = start
    tourDist = 0
    cityDist = cityDistance(cities)
    while unvisited:
        # 当unvisited集合不为空时，重复循环
        # 找到当前位置在unvisited中的最近邻
        nextToVisit, dist = nearestNeighbor(cityDist, currentPos, unvisited)
        tourDist += dist
        currentPos = nextToVisit
        tour.append(nextToVisit)
        unvisited.remove(nextToVisit)
    # 重新回到起点
    tour.append(start)
    tourDist += cityDist[currentPos, start]
    return tour,tourDist
nCities = 30
cities = genCity(nCities) # 随机生成nCities个城市坐标
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
from scipy.spatial import distance
from deap import creator, base, tools, algorithms
import random
nCities = 30
cities = genCity(nCities) # 随机生成nCities个城市坐标

tour,tourDist = nnTSP(cities)
print('nnTSP寻找到的最优路径为：' + str(tour))
print('nnTSP寻找到最优路径长度为：' + str(tourDist))
plotTour(tour, cities)

runfile('C:/Users/92156/.spyder-py3/小批量随机梯度.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/TSP.py', wdir='C:/Users/92156/.spyder-py3')
repnnTSPtourOptimized, repnnTSPtourDistOptimized = opt(cityDist, repnnTSPtour, 2)
print('repnnTSPtour + 2OPT优化后的最优路径为：' + str(repnnTSPtourOptimized))
print('repnnTSPtour + 2OPT优化后的最优路径长度为：' + str(repnnTSPtourDistOptimized))
plotTour(repnnTSPtourOptimized, cities)
optimizedRoute, minDistance = opt(cityDist, tour)
print('nnTSP + 2OPT优化后的最优路径为：' + str(optimizedRoute))
print('nnTSP + 2OPT优化后的最优路径长度为：' + str(minDistance))
plotTour(optimizedRoute, cities)
runfile('C:/Users/92156/.spyder-py3/TSP.py', wdir='C:/Users/92156/.spyder-py3')
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
from scipy.spatial import distance
from deap import creator, base, tools, algorithms
import random

params = {
    'font.family': 'serif',
    'figure.figsize': [4.0, 3.0],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'legend.fontsize': 'small'
}
plt.rcParams.update(params)
#--------------------------------
## 定义TSP中的基本元素

# 用[n,2]的np.array存储城市坐标；每行存储一个城市
def genCity(n, Lb = 100 ,Ub = 999):
    # 生成城市坐标
    # 输入：n -- 需要生成的城市数量
    # 输出: nx2 np array 每行是一个城市的[X,Y]坐标
    np.random.seed(42) # 保证结果的可复现性
    return np.random.randint(low = Lb, high = Ub, size=(n,2))


# 计算并存储城市距离矩阵
def cityDistance(cities):
    # 生成城市距离矩阵 distMat[A,B] = distMat[B,A]表示城市A，B之间距离
    # 输入：cities -- [n,2] np array， 表示城市坐标
    # 输出：nxn np array， 存储城市两两之间的距离
    return distance.cdist(cities, cities, 'euclidean')


def completeRoute(individual):
    # 序列编码时，缺少最后一段回到原点的线段
    return individual + [individual[0]] # 不要用append


# 计算给定路线的长度
def routeDistance(route):
    # 输入：
    #      route -- 一条路线，一个sequence
    # 输出：routeDist -- scalar，路线的长度
    if route[0] != route[-1]:
        route = completeRoute(route)
    routeDist = 0
    for i,j in zip(route[0::],route[1::]):
        routeDist += cityDist[i,j] # 这里直接从cityDist变量中取值了，其实并不是很安全的写法，单纯偷懒了
    return (routeDist), # 注意DEAP要求评价函数返回一个元组


# 路径可视化
def plotTour(tour, cities, style = 'bo-'):
    if len(tour)>1000: plt.figure(figsize = (15,10))
    start = tour[0:1]
    for i,j in zip(tour[0::], tour[1::]):
        plt.plot([cities[i,0],cities[j,0]], [cities[i,1],cities[j,1]], style)
    plt.plot(cities[start,0],cities[start,1],'rD')
    plt.axis('scaled')
    plt.axis('off')


def nearestNeighbor(cityDist, currentPos, unvisited):
    # 输入：cityDist -- [n,n] np array,记录城市间距离
    # currentPos -- 一个数字，指示当前位于的城市
    # unvisited -- 一个列表，指示可选邻居列表
    # 输出：
    # nextToVisit -- 一个数字，给出最近邻的index
    # dist -- 一个数字，给出到最近邻的距离
    neighborDist = cityDist[currentPos,unvisited]
    neighborIdx = np.argmin(neighborDist)
    nextToVisit = unvisited[neighborIdx]
    dist = neighborDist[neighborIdx]
    return nextToVisit, dist


# 用贪婪算法求解TSP
def nnTSP(cities, start = 0):
    cityList = list(range(cities.shape[0]))
    tour = [start]
    unvisited = cityList.copy()
    unvisited.remove(start)
    currentPos = start
    tourDist = 0
    cityDist = cityDistance(cities)
    while unvisited:
        # 当unvisited集合不为空时，重复循环
        # 找到当前位置在unvisited中的最近邻
        nextToVisit, dist = nearestNeighbor(cityDist, currentPos, unvisited)
        tourDist += dist
        currentPos = nextToVisit
        tour.append(nextToVisit)
        unvisited.remove(nextToVisit)
    # 重新回到起点
    tour.append(start)
    tourDist += cityDist[currentPos, start]
    return tour,tourDist

def repNNTSP(cities):
    optimizedRoute = [] # 最优路径
    minDistance = np.Inf # 最优路径长度
    for i in range(len(cities)):
        tour,tourDist = nnTSP(cities, start = i)
        if tourDist < minDistance:
            optimizedRoute = tour
            minDistance = tourDist
    return optimizedRoute, minDistance

def opt(cityDist, route, k=2):
    # 用2-opt算法优化路径
    # 输入：cityDist -- [n,n]矩阵，记录了城市间的距离
    # route -- sequence，记录路径
    # 输出： 优化后的路径optimizedRoute及其路径长度
    nCities = len(route) # 城市数
    optimizedRoute = route # 最优路径
    minDistance = routeDistance(route) # 最优路径长度
    for i in range(1,nCities-2):
        for j in range(i+k, nCities):
            if j-i == 1:
                continue
            reversedRoute = route[:i]+route[i:j][::-1]+route[j:]# 翻转后的路径
            reversedRouteDist = routeDistance(reversedRoute)
            # 如果翻转后路径更优，则更新最优解
            if  reversedRouteDist < minDistance:
                minDistance = reversedRouteDist
                optimizedRoute = reversedRoute
    return optimizedRoute, minDistance

def reopt(cities):
    optimizedRoute = [] # 最优路径
    minDistance = np.Inf # 最优路径长度
    for i in range(len(cities)):
        tour,tourDist = opt(cities, start = i)
        if tourDist < minDistance:
            optimizedRoute = tour
            minDistance = tourDist
    return optimizedRoute, minDistance

#--------------------------------
## 设计GA算法
nCities = 30
cities = genCity(nCities) # 随机生成nCities个城市坐标

# 问题定义
creator.create('FitnessMin', base.Fitness, weights=(-1.0,)) # 最小化问题
creator.create('Individual', list, fitness=creator.FitnessMin)

# 定义个体编码
toolbox = base.Toolbox()
toolbox.register('indices', random.sample, range(nCities), nCities) # 创建序列
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indices)

# 生成族群
N_POP = 1000
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(N_POP)

# 注册所需工具
cityDist = cityDistance(cities)
toolbox.register('evaluate', routeDistance)
toolbox.register('select', tools.selTournament, tournsize = 2)
toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.2)

# 数据记录
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('min', np.min)

# 调用内置的进化算法
resultPop, logbook = algorithms.eaSimple(pop, toolbox, cxpb = 0.5, mutpb = 0.2, ngen = 100, stats = stats, verbose = True)
#ngen  是迭代次数
tour = tools.selBest(resultPop, k=1)[0]
tourDist = tour.fitness
tour = completeRoute(tour)
print('遗传算法最优路径为:'+str(tour))
print('最优路径距离为：'+str(tourDist))
plotTour(tour, cities)
tour,tourDist = repNNTSP(cities)
print('rennTSP寻找到的最优路径为：' + str(tour))
print('nnTSP寻找到最优路径长度为：' + str(tourDist))
plotTour(tour, cities)
optimizedRoute, minDistance = opt(cityDist, tour)
print('nnTSP + 2OPT优化后的最优路径为：' + str(optimizedRoute))
print('nnTSP + 2OPT优化后的最优路径长度为：' + str(minDistance))
plotTour(optimizedRoute, cities)
%reset
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
from scipy.spatial import distance
from deap import creator, base, tools, algorithms
import random

params = {
    'font.family': 'serif',
    'figure.figsize': [4.0, 3.0],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'legend.fontsize': 'small'
}
plt.rcParams.update(params)
#--------------------------------
## 定义TSP中的基本元素

# 用[n,2]的np.array存储城市坐标；每行存储一个城市
def genCity(n, Lb = 100 ,Ub = 999):
    # 生成城市坐标
    # 输入：n -- 需要生成的城市数量
    # 输出: nx2 np array 每行是一个城市的[X,Y]坐标
    np.random.seed(42) # 保证结果的可复现性
    return np.random.randint(low = Lb, high = Ub, size=(n,2))


# 计算并存储城市距离矩阵
def cityDistance(cities):
    # 生成城市距离矩阵 distMat[A,B] = distMat[B,A]表示城市A，B之间距离
    # 输入：cities -- [n,2] np array， 表示城市坐标
    # 输出：nxn np array， 存储城市两两之间的距离
    return distance.cdist(cities, cities, 'euclidean')


def completeRoute(individual):
    # 序列编码时，缺少最后一段回到原点的线段
    return individual + [individual[0]] # 不要用append


# 计算给定路线的长度
def routeDistance(route):
    # 输入：
    #      route -- 一条路线，一个sequence
    # 输出：routeDist -- scalar，路线的长度
    if route[0] != route[-1]:
        route = completeRoute(route)
    routeDist = 0
    for i,j in zip(route[0::],route[1::]):
        routeDist += cityDist[i,j] # 这里直接从cityDist变量中取值了，其实并不是很安全的写法，单纯偷懒了
    return (routeDist), # 注意DEAP要求评价函数返回一个元组


# 路径可视化
def plotTour(tour, cities, style = 'bo-'):
    if len(tour)>1000: plt.figure(figsize = (15,10))
    start = tour[0:1]
    for i,j in zip(tour[0::], tour[1::]):
        plt.plot([cities[i,0],cities[j,0]], [cities[i,1],cities[j,1]], style)
    plt.plot(cities[start,0],cities[start,1],'rD')
    plt.axis('scaled')
    plt.axis('off')


def nearestNeighbor(cityDist, currentPos, unvisited):
    # 输入：cityDist -- [n,n] np array,记录城市间距离
    # currentPos -- 一个数字，指示当前位于的城市
    # unvisited -- 一个列表，指示可选邻居列表
    # 输出：
    # nextToVisit -- 一个数字，给出最近邻的index
    # dist -- 一个数字，给出到最近邻的距离
    neighborDist = cityDist[currentPos,unvisited]
    neighborIdx = np.argmin(neighborDist)
    nextToVisit = unvisited[neighborIdx]
    dist = neighborDist[neighborIdx]
    return nextToVisit, dist


# 用贪婪算法求解TSP
def nnTSP(cities, start = 0):
    cityList = list(range(cities.shape[0]))
    tour = [start]
    unvisited = cityList.copy()
    unvisited.remove(start)
    currentPos = start
    tourDist = 0
    cityDist = cityDistance(cities)
    while unvisited:
        # 当unvisited集合不为空时，重复循环
        # 找到当前位置在unvisited中的最近邻
        nextToVisit, dist = nearestNeighbor(cityDist, currentPos, unvisited)
        tourDist += dist
        currentPos = nextToVisit
        tour.append(nextToVisit)
        unvisited.remove(nextToVisit)
    # 重新回到起点
    tour.append(start)
    tourDist += cityDist[currentPos, start]
    return tour,tourDist

def repNNTSP(cities):
    optimizedRoute = [] # 最优路径
    minDistance = np.Inf # 最优路径长度
    for i in range(len(cities)):
        tour,tourDist = nnTSP(cities, start = i)
        if tourDist < minDistance:
            optimizedRoute = tour
            minDistance = tourDist
    return optimizedRoute, minDistance

def opt(cityDist, route, k=2):
    # 用2-opt算法优化路径
    # 输入：cityDist -- [n,n]矩阵，记录了城市间的距离
    # route -- sequence，记录路径
    # 输出： 优化后的路径optimizedRoute及其路径长度
    nCities = len(route) # 城市数
    optimizedRoute = route # 最优路径
    minDistance = routeDistance(route) # 最优路径长度
    for i in range(1,nCities-2):
        for j in range(i+k, nCities):
            if j-i == 1:
                continue
            reversedRoute = route[:i]+route[i:j][::-1]+route[j:]# 翻转后的路径
            reversedRouteDist = routeDistance(reversedRoute)
            # 如果翻转后路径更优，则更新最优解
            if  reversedRouteDist < minDistance:
                minDistance = reversedRouteDist
                optimizedRoute = reversedRoute
    return optimizedRoute, minDistance

def reopt(cities):
    optimizedRoute = [] # 最优路径
    minDistance = np.Inf # 最优路径长度
    for i in range(len(cities)):
        tour,tourDist = opt(cities, start = i)
        if tourDist < minDistance:
            optimizedRoute = tour
            minDistance = tourDist
    return optimizedRoute, minDistance

#--------------------------------
## 设计GA算法
nCities = 30
cities = genCity(nCities) # 随机生成nCities个城市坐标

# 问题定义
creator.create('FitnessMin', base.Fitness, weights=(-1.0,)) # 最小化问题
creator.create('Individual', list, fitness=creator.FitnessMin)

# 定义个体编码
toolbox = base.Toolbox()
toolbox.register('indices', random.sample, range(nCities), nCities) # 创建序列
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indices)

# 生成族群
N_POP = 1000
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(N_POP)

# 注册所需工具
cityDist = cityDistance(cities)
toolbox.register('evaluate', routeDistance)
toolbox.register('select', tools.selTournament, tournsize = 2)
toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.2)

# 数据记录
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('min', np.min)

# 调用内置的进化算法
resultPop, logbook = algorithms.eaSimple(pop, toolbox, cxpb = 0.5, mutpb = 0.2, ngen = 100, stats = stats, verbose = True)
#ngen  是迭代次数
tour = tools.selBest(resultPop, k=1)[0]
tourDist = tour.fitness
tour = completeRoute(tour)
print('遗传算法最优路径为:'+str(tour))
print('最优路径距离为：'+str(tourDist))
plotTour(tour, cities)
tour,tourDist = repNNTSP(cities)
print('rennTSP寻找到的最优路径为：' + str(tour))
print('nnTSP寻找到最优路径长度为：' + str(tourDist))
plotTour(tour, cities)
repNNTSP=tour
repnnTSPtour=tour
optimizedRoute, minDistance = opt(cityDist, tour)
print('nnTSP + 2OPT优化后的最优路径为：' + str(optimizedRoute))
print('nnTSP + 2OPT优化后的最优路径长度为：' + str(minDistance))
plotTour(optimizedRoute, cities)
repnnTSPtourOptimized, repnnTSPtourDistOptimized = opt(cityDist, repnnTSPtour, 2)
print('repnnTSPtour + 2OPT优化后的最优路径为：' + str(repnnTSPtourOptimized))
print('repnnTSPtour + 2OPT优化后的最优路径长度为：' + str(repnnTSPtourDistOptimized))
plotTour(repnnTSPtourOptimized, cities)
runfile('C:/Users/92156/.spyder-py3/TSP.py', wdir='C:/Users/92156/.spyder-py3')

## ---(Sun Sep  8 08:49:55 2019)---
import random
import operator
import numpy as np
import matplotlib.pyplot as plt

class Road():
	def __init__(self,start_x=0,start_y=9,end_x=9,end_y=0,errorList=[]):
		self.start_x = start_x
		self.start_y = start_y
		self.end_x = end_x
		self.end_y = end_y
		self.errorList = errorList
		self.road = []
		self.x = 0
		self.y = 0
		self.step = 0

	def Option(self):
		self.x = self.start_x
		self.y = self.start_y
		self.road.append((self.x,self.y))


		while True:
			self.x = self.road[-1][0]
			self.y = self.road[-1][1]


			#判定机器人走的初始方向
			if self.randNum(0,9) in [0,1,2]:				
				if (self.x+1,self.y) not in self.errorList:
					self.x += 1					
					self.road.append((self.x,self.y))

			elif self.randNum(0,9) in [3,4,5]:				
				if (self.x, self.y-1) not in self.errorList:
					self.y-=1					
					self.road.append((self.x,self.y))

			elif self.randNum(0,9) in [6,7]:				
				if (self.x-1, self.y) not in self.errorList:
					self.x -= 1					
					self.road.append((self.x,self.y))

			elif self.randNum(0,9) in [8,9]:				
				if (self.x, self.y+1) not in self.errorList:
					self.y += 1					
					self.road.append((self.x,self.y))

			#判定机器人是否卡死
			if ((self.x+1,self.y)in self.errorList and (self.x,self.y-1) in self.errorList):				
				if self.randNum(0,1) == 0:					
					if (self.x-1, self.y) not in self.errorList:
						self.x -= 1						
						self.road.append((self.x,self.y)) 
				elif self.randNum(0,1) == 1:					
					if (self.x,self.y+1) not in self.errorList:
						self.y += 1 						
						self.road.append((self.x,self.y))


			#判定机器人是否越界
			if not(-1<self.road[-1][0]<10 and -1<self.road[-1][1]<10):				
				self.road.pop()

			#判定是否重复
			if len(self.road)>3:
				if self.road[-1]==self.road[-3]:
					self.road.pop()
					self.road.pop()

			#判定机器人是否到终点
			if (self.x,self.y) == (self.end_x,self.end_y):				
				break



	def getStep(self):
		self.step = len(self.road)
		return self.step

	def getRoad(self):
		return self.road

	def randNum(self,a,b):
		r = random.randint(a,b)
		return r

def main(turn):
	errorList = [(3,9),\
	(0,8),(2,8),(3,8),(6,8),(8,8),\
	(2,7),(6,7),(7,7),\
	(1,5),(2,5),(3,5),(6,5),(8,5),(9,5),\
	(5,4),(6,4),(9,4),\
	(2,3),(3,3),(9,3),\
	(3,2),(2,2),(6,2),(7,2),\
	(1,1),(5,1),(6,1),\
	(1,0),(8,0)]

	to ={}
	tb = []	
	i=0
	while i<turn:
		print('第%d次实验'%(i+1))
		road = Road(0,9,9,0,errorList)
		road.Option()
		step = road.getStep()
		print("步长:%d"%step)
		position = road.getRoad()
		print('路径:\n:'+str(position))
		to['step'] = step
		to['position'] = position
		tb.append(to)
		i += 1
	tb = sorted(tb,key=operator.itemgetter('step'))
	print('最优步长%d'%tb[0]['step'])
	print('最有路径:\n'+str(tb[0]['position']))
	pl(tb[0]['position'])

def pl(tb):
	x=[]
	y=[]
	for i in tb:
		x.append(i[0])
		y.append(i[1])

	#创建绘图对象，figsize参数可以指定绘图对象的宽度和高度，单位为英寸，一英寸=80px
	plt.figure(figsize=(8,8))
	#在当前绘图对象进行绘图（两个参数是x,y轴的数据）
	plt.plot(x, y, linewidth = '1', label ="test",color='#054E9F', linestyle=':', marker='|')
	plt.xlabel("X") #X轴标签
	plt.ylabel("Y") #Y轴标签
	plt.title("机器人运动方向") #标题
	#保存图象
	# plt.savefig("easyplot.png")
	plt.legend()
	plt.show()

main(100)



import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

EXTEND_AREA = 10.0  # [m] grid map extention length

show_animation = True

def generate_gaussian_grid_map(ox, oy, xyreso, std):
    minx, miny, maxx, maxy, xw, yw = calc_grid_map_config(ox, oy, xyreso)
    gmap = [[0.0 for i in range(yw)] for i in range(xw)]
    for ix in range(xw):
        for iy in range(yw):
            x = ix * xyreso + minx
            y = iy * xyreso + miny
            #Search minimum distance
            mindis = float("inf")
            for (iox, ioy) in zip(ox, oy):
                d = math.sqrt((iox - x)* * 2 + (ioy - y)* * 2)
                if mindis >= d:
                    mindis = d
            pdf = (1.0 - norm.cdf(mindis, 0.0, std))
            gmap[ix][iy] = pdf
    return gmap, minx, maxx, miny, maxy

def calc_grid_map_config(ox, oy, xyreso):
    minx = round(min(ox) - EXTEND_AREA / 2.0)
    miny = round(min(oy) - EXTEND_AREA / 2.0)
    maxx = round(max(ox) + EXTEND_AREA / 2.0)
    maxy = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))
    return minx, miny, maxx, maxy, xw, yw
def generate_gaussian_grid_map(ox, oy, xyreso, std):
    minx, miny, maxx, maxy, xw, yw = calc_grid_map_config(ox, oy, xyreso)
    gmap = [[0.0 for i in range(yw)] for i in range(xw)]
    for ix in range(xw):
        for iy in range(yw):
            x = ix * xyreso + minx
            y = iy * xyreso + miny
            #Search minimum distance
            mindis = float("inf")
            for (iox, ioy) in zip(ox, oy):
                d = math.sqrt((iox - x)** 2 + (ioy - y)** 2)
                if mindis >= d:
                    mindis = d
            pdf = (1.0 - norm.cdf(mindis, 0.0, std))
            gmap[ix][iy] = pdf
    return gmap, minx, maxx, miny, maxy

def calc_grid_map_config(ox, oy, xyreso):
    minx = round(min(ox) - EXTEND_AREA / 2.0)
    miny = round(min(oy) - EXTEND_AREA / 2.0)
    maxx = round(max(ox) + EXTEND_AREA / 2.0)
    maxy = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))
    return minx, miny, maxx, maxy, xw, yw

def draw_heatmap(data, minx, maxx, miny, maxy, xyreso):
    x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso),
                    slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)]
    plt.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues)
    plt.axis("equal")

def main():
    print(_ _file__ + " start!!")
    xyreso = 0.5  # xy grid resolution
    STD = 5.0  # standard diviation for gaussian distribution
    for i in range(5):
        ox = (np.random.rand(4) - 0.5) * 10.0
        oy = (np.random.rand(4) - 0.5) * 10.0
        gmap, minx, maxx, miny, maxy = generate_gaussian_grid_map(ox, oy,
                                                                 xyreso, STD)
        if show_animation:
            plt.cla()
            draw_heatmap(gmap, minx, maxx, miny, maxy, xyreso)
            plt.plot(ox, oy, "xr")
            plt.plot(0.0, 0.0, "ob")
            plt.pause(1.0)

if _ _ name__ == '_ _main__':
    main()
def draw_heatmap(data, minx, maxx, miny, maxy, xyreso):
    x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso),
                    slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)]
    plt.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues)
    plt.axis("equal")

def main():
    print(__file__ + " start!!")
    xyreso = 0.5  # xy grid resolution
    STD = 5.0  # standard diviation for gaussian distribution
    for i in range(5):
        ox = (np.random.rand(4) - 0.5) * 10.0
        oy = (np.random.rand(4) - 0.5) * 10.0
        gmap, minx, maxx, miny, maxy = generate_gaussian_grid_map(ox, oy,
                                                                 xyreso, STD)
        if show_animation:
            plt.cla()
            draw_heatmap(gmap, minx, maxx, miny, maxy, xyreso)
            plt.plot(ox, oy, "xr")
            plt.plot(0.0, 0.0, "ob")
            plt.pause(1.0)

if _ _ name__ == '_ _main__':
    main()
def draw_heatmap(data, minx, maxx, miny, maxy, xyreso):
    x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso),
                    slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)]
    plt.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues)
    plt.axis("equal")

def main():
    print(__file__ + " start!!")
    xyreso = 0.5  # xy grid resolution
    STD = 5.0  # standard diviation for gaussian distribution
    for i in range(5):
        ox = (np.random.rand(4) - 0.5) * 10.0
        oy = (np.random.rand(4) - 0.5) * 10.0
        gmap, minx, maxx, miny, maxy = generate_gaussian_grid_map(ox, oy,
                                                                 xyreso, STD)
        if show_animation:
            plt.cla()
            draw_heatmap(gmap, minx, maxx, miny, maxy, xyreso)
            plt.plot(ox, oy, "xr")
            plt.plot(0.0, 0.0, "ob")
            plt.pause(1.0)

if __name__ == '__main__':
    main()
    
if "main" == "main":
    main()
runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/test.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')
import random
import operator
import numpy as np
import matplotlib.pyplot as plt

class Road():
	def __init__(self,start_x=0,start_y=9,end_x=9,end_y=0,errorList=[]):
		self.start_x = start_x
		self.start_y = start_y
		self.end_x = end_x
		self.end_y = end_y
		self.errorList = errorList
		self.road = []
		self.x = 0
		self.y = 0
		self.step = 0
 
	def Option(self):
		self.x = self.start_x
		self.y = self.start_y
		self.road.append((self.x,self.y))
  
  
		while True:
			self.x = self.road[-1][0]
			self.y = self.road[-1][1]
   
   
			#判定机器人走的初始方向
			if self.randNum(0,9) in [0,1,2]:				
				if (self.x+1,self.y) not in self.errorList:
					self.x += 1					
					self.road.append((self.x,self.y))
   
			elif self.randNum(0,9) in [3,4,5]:				
				if (self.x, self.y-1) not in self.errorList:
					self.y-=1					
					self.road.append((self.x,self.y))
   
			elif self.randNum(0,9) in [6,7]:				
				if (self.x-1, self.y) not in self.errorList:
					self.x -= 1					
					self.road.append((self.x,self.y))
   
			elif self.randNum(0,9) in [8,9]:				
				if (self.x, self.y+1) not in self.errorList:
					self.y += 1					
					self.road.append((self.x,self.y))
   
			#判定机器人是否卡死
			if ((self.x+1,self.y)in self.errorList and (self.x,self.y-1) in self.errorList):				
				if self.randNum(0,1) == 0:					
					if (self.x-1, self.y) not in self.errorList:
						self.x -= 1						
						self.road.append((self.x,self.y)) 
				elif self.randNum(0,1) == 1:					
					if (self.x,self.y+1) not in self.errorList:
						self.y += 1 						
						self.road.append((self.x,self.y))
   
   
			#判定机器人是否越界
			if not(-1<self.road[-1][0]<10 and -1<self.road[-1][1]<10):				
				self.road.pop()
   
			#判定是否重复
			if len(self.road)>3:
				if self.road[-1]==self.road[-3]:
					self.road.pop()
					self.road.pop()
   
			#判定机器人是否到终点
			if (self.x,self.y) == (self.end_x,self.end_y):				
				break
 
 
 
	def getStep(self):
		self.step = len(self.road)
		return self.step
 
	def getRoad(self):
		return self.road
 
	def randNum(self,a,b):
		r = random.randint(a,b)
		return r
runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/test.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')
"""
Grid based Dijkstra planning
author: Atsushi Sakai(@Atsushi_twi)
"""

import matplotlib.pyplot as plt
import math

show_animation = True

class Dijkstra:

    def __init__(self, ox, oy, reso, rr):
        """
        Initialize map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        reso: grid resolution [m]
        rr: robot radius[m]
        """

        self.reso = reso
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, pind):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.pind = pind

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

    def planning(self, sx, sy, gx, gy):
        """
        dijkstra path search
        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        nstart = self.Node(self.calc_xyindex(sx, self.minx),
                           self.calc_xyindex(sy, self.miny), 0.0, -1)
        ngoal = self.Node(self.calc_xyindex(gx, self.minx),
                          self.calc_xyindex(gy, self.miny), 0.0, -1)

        openset, closedset = dict(), dict()
        openset[self.calc_index(nstart)] = nstart

        while 1:
            c_id = min(openset, key=lambda o: openset[o].cost)
            current = openset[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_position(current.x, self.minx),
                         self.calc_position(current.y, self.miny), "xc")
                if len(closedset.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == ngoal.x and current.y == ngoal.y:
                print("Find goal")
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            # Remove the item from the open set
            del openset[c_id]

            # Add it to the closed set
            closedset[c_id] = current

            # expand search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_index(node)

                if n_id in closedset:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in openset:
                    openset[n_id] = node  # Discover a new node
                else:
                    if openset[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        openset[n_id] = node

        rx, ry = self.calc_final_path(ngoal, closedset)

        return rx, ry

    def calc_final_path(self, ngoal, closedset):
        # generate final course
        rx, ry = [self.calc_position(ngoal.x, self.minx)], [
            self.calc_position(ngoal.y, self.miny)]
        pind = ngoal.pind
        while pind != -1:
            n = closedset[pind]
            rx.append(self.calc_position(n.x, self.minx))
            ry.append(self.calc_position(n.y, self.miny))
            pind = n.pind

        return rx, ry

    def calc_heuristic(self, n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
        return d

    def calc_position(self, index, minp):
        pos = index*self.reso+minp
        return pos

    def calc_xyindex(self, position, minp):
        return round((position - minp)/self.reso)

    def calc_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def verify_node(self, node):
        px = self.calc_position(node.x, self.minx)
        py = self.calc_position(node.y, self.miny)

        if px < self.minx:
            return False
        elif py < self.miny:
            return False
        elif px >= self.maxx:
            return False
        elif py >= self.maxy:
            return False

        if self.obmap[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.minx = round(min(ox))
        self.miny = round(min(oy))
        self.maxx = round(max(ox))
        self.maxy = round(max(oy))
        print("minx:", self.minx)
        print("miny:", self.miny)
        print("maxx:", self.maxx)
        print("maxy:", self.maxy)

        self.xwidth = round((self.maxx - self.minx)/self.reso)
        self.ywidth = round((self.maxy - self.miny)/self.reso)
        print("xwidth:", self.xwidth)
        print("ywidth:", self.ywidth)

        # obstacle map generation
        self.obmap = [[False for i in range(self.ywidth)]
                      for i in range(self.xwidth)]
        for ix in range(self.xwidth):
            x = self.calc_position(ix, self.minx)
            for iy in range(self.ywidth):
                y = self.calc_position(iy, self.miny)
                for iox, ioy in zip(ox, oy):
                    d = math.sqrt((iox - x)**2 + (ioy - y)**2)
                    if d <= self.rr:
                        self.obmap[ix][iy] = True
                        break

    def get_motion_model(self):
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = -5.0  # [m]
    sy = -5.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    dijkstra = Dijkstra(ox, oy, grid_size, robot_radius)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()


__file__=[]
"""
Grid based Dijkstra planning
author: Atsushi Sakai(@Atsushi_twi)
"""

import matplotlib.pyplot as plt
import math

show_animation = True

class Dijkstra:

    def __init__(self, ox, oy, reso, rr):
        """
        Initialize map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        reso: grid resolution [m]
        rr: robot radius[m]
        """

        self.reso = reso
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, pind):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.pind = pind

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

    def planning(self, sx, sy, gx, gy):
        """
        dijkstra path search
        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        nstart = self.Node(self.calc_xyindex(sx, self.minx),
                           self.calc_xyindex(sy, self.miny), 0.0, -1)
        ngoal = self.Node(self.calc_xyindex(gx, self.minx),
                          self.calc_xyindex(gy, self.miny), 0.0, -1)

        openset, closedset = dict(), dict()
        openset[self.calc_index(nstart)] = nstart

        while 1:
            c_id = min(openset, key=lambda o: openset[o].cost)
            current = openset[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_position(current.x, self.minx),
                         self.calc_position(current.y, self.miny), "xc")
                if len(closedset.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == ngoal.x and current.y == ngoal.y:
                print("Find goal")
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            # Remove the item from the open set
            del openset[c_id]

            # Add it to the closed set
            closedset[c_id] = current

            # expand search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_index(node)

                if n_id in closedset:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in openset:
                    openset[n_id] = node  # Discover a new node
                else:
                    if openset[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        openset[n_id] = node

        rx, ry = self.calc_final_path(ngoal, closedset)

        return rx, ry

    def calc_final_path(self, ngoal, closedset):
        # generate final course
        rx, ry = [self.calc_position(ngoal.x, self.minx)], [
            self.calc_position(ngoal.y, self.miny)]
        pind = ngoal.pind
        while pind != -1:
            n = closedset[pind]
            rx.append(self.calc_position(n.x, self.minx))
            ry.append(self.calc_position(n.y, self.miny))
            pind = n.pind

        return rx, ry

    def calc_heuristic(self, n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
        return d

    def calc_position(self, index, minp):
        pos = index*self.reso+minp
        return pos

    def calc_xyindex(self, position, minp):
        return round((position - minp)/self.reso)

    def calc_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def verify_node(self, node):
        px = self.calc_position(node.x, self.minx)
        py = self.calc_position(node.y, self.miny)

        if px < self.minx:
            return False
        elif py < self.miny:
            return False
        elif px >= self.maxx:
            return False
        elif py >= self.maxy:
            return False

        if self.obmap[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.minx = round(min(ox))
        self.miny = round(min(oy))
        self.maxx = round(max(ox))
        self.maxy = round(max(oy))
        print("minx:", self.minx)
        print("miny:", self.miny)
        print("maxx:", self.maxx)
        print("maxy:", self.maxy)

        self.xwidth = round((self.maxx - self.minx)/self.reso)
        self.ywidth = round((self.maxy - self.miny)/self.reso)
        print("xwidth:", self.xwidth)
        print("ywidth:", self.ywidth)

        # obstacle map generation
        self.obmap = [[False for i in range(self.ywidth)]
                      for i in range(self.xwidth)]
        for ix in range(self.xwidth):
            x = self.calc_position(ix, self.minx)
            for iy in range(self.ywidth):
                y = self.calc_position(iy, self.miny)
                for iox, ioy in zip(ox, oy):
                    d = math.sqrt((iox - x)**2 + (ioy - y)**2)
                    if d <= self.rr:
                        self.obmap[ix][iy] = True
                        break

    def get_motion_model(self):
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = -5.0  # [m]
    sy = -5.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    dijkstra = Dijkstra(ox, oy, grid_size, robot_radius)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()

__file__=""
"""
Grid based Dijkstra planning
author: Atsushi Sakai(@Atsushi_twi)
"""

import matplotlib.pyplot as plt
import math

show_animation = True

class Dijkstra:

    def __init__(self, ox, oy, reso, rr):
        """
        Initialize map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        reso: grid resolution [m]
        rr: robot radius[m]
        """

        self.reso = reso
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, pind):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.pind = pind

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

    def planning(self, sx, sy, gx, gy):
        """
        dijkstra path search
        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        nstart = self.Node(self.calc_xyindex(sx, self.minx),
                           self.calc_xyindex(sy, self.miny), 0.0, -1)
        ngoal = self.Node(self.calc_xyindex(gx, self.minx),
                          self.calc_xyindex(gy, self.miny), 0.0, -1)

        openset, closedset = dict(), dict()
        openset[self.calc_index(nstart)] = nstart

        while 1:
            c_id = min(openset, key=lambda o: openset[o].cost)
            current = openset[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_position(current.x, self.minx),
                         self.calc_position(current.y, self.miny), "xc")
                if len(closedset.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == ngoal.x and current.y == ngoal.y:
                print("Find goal")
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            # Remove the item from the open set
            del openset[c_id]

            # Add it to the closed set
            closedset[c_id] = current

            # expand search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_index(node)

                if n_id in closedset:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in openset:
                    openset[n_id] = node  # Discover a new node
                else:
                    if openset[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        openset[n_id] = node

        rx, ry = self.calc_final_path(ngoal, closedset)

        return rx, ry

    def calc_final_path(self, ngoal, closedset):
        # generate final course
        rx, ry = [self.calc_position(ngoal.x, self.minx)], [
            self.calc_position(ngoal.y, self.miny)]
        pind = ngoal.pind
        while pind != -1:
            n = closedset[pind]
            rx.append(self.calc_position(n.x, self.minx))
            ry.append(self.calc_position(n.y, self.miny))
            pind = n.pind

        return rx, ry

    def calc_heuristic(self, n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
        return d

    def calc_position(self, index, minp):
        pos = index*self.reso+minp
        return pos

    def calc_xyindex(self, position, minp):
        return round((position - minp)/self.reso)

    def calc_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def verify_node(self, node):
        px = self.calc_position(node.x, self.minx)
        py = self.calc_position(node.y, self.miny)

        if px < self.minx:
            return False
        elif py < self.miny:
            return False
        elif px >= self.maxx:
            return False
        elif py >= self.maxy:
            return False

        if self.obmap[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.minx = round(min(ox))
        self.miny = round(min(oy))
        self.maxx = round(max(ox))
        self.maxy = round(max(oy))
        print("minx:", self.minx)
        print("miny:", self.miny)
        print("maxx:", self.maxx)
        print("maxy:", self.maxy)

        self.xwidth = round((self.maxx - self.minx)/self.reso)
        self.ywidth = round((self.maxy - self.miny)/self.reso)
        print("xwidth:", self.xwidth)
        print("ywidth:", self.ywidth)

        # obstacle map generation
        self.obmap = [[False for i in range(self.ywidth)]
                      for i in range(self.xwidth)]
        for ix in range(self.xwidth):
            x = self.calc_position(ix, self.minx)
            for iy in range(self.ywidth):
                y = self.calc_position(iy, self.miny)
                for iox, ioy in zip(ox, oy):
                    d = math.sqrt((iox - x)**2 + (ioy - y)**2)
                    if d <= self.rr:
                        self.obmap[ix][iy] = True
                        break

    def get_motion_model(self):
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = -5.0  # [m]
    sy = -5.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    dijkstra = Dijkstra(ox, oy, grid_size, robot_radius)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()

"""
Grid based Dijkstra planning
author: Atsushi Sakai(@Atsushi_twi)
"""

import matplotlib.pyplot as plt
import math

show_animation = True

class Dijkstra:

    def __init__(self, ox, oy, reso, rr):
        """
        Initialize map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        reso: grid resolution [m]
        rr: robot radius[m]
        """

        self.reso = reso
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, pind):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.pind = pind

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

    def planning(self, sx, sy, gx, gy):
        """
        dijkstra path search
        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        nstart = self.Node(self.calc_xyindex(sx, self.minx),
                           self.calc_xyindex(sy, self.miny), 0.0, -1)
        ngoal = self.Node(self.calc_xyindex(gx, self.minx),
                          self.calc_xyindex(gy, self.miny), 0.0, -1)

        openset, closedset = dict(), dict()
        openset[self.calc_index(nstart)] = nstart

        while 1:
            c_id = min(openset, key=lambda o: openset[o].cost)
            current = openset[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_position(current.x, self.minx),
                         self.calc_position(current.y, self.miny), "xc")
                if len(closedset.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == ngoal.x and current.y == ngoal.y:
                print("Find goal")
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            # Remove the item from the open set
            del openset[c_id]

            # Add it to the closed set
            closedset[c_id] = current

            # expand search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_index(node)

                if n_id in closedset:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in openset:
                    openset[n_id] = node  # Discover a new node
                else:
                    if openset[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        openset[n_id] = node

        rx, ry = self.calc_final_path(ngoal, closedset)

        return rx, ry

    def calc_final_path(self, ngoal, closedset):
        # generate final course
        rx, ry = [self.calc_position(ngoal.x, self.minx)], [
            self.calc_position(ngoal.y, self.miny)]
        pind = ngoal.pind
        while pind != -1:
            n = closedset[pind]
            rx.append(self.calc_position(n.x, self.minx))
            ry.append(self.calc_position(n.y, self.miny))
            pind = n.pind

        return rx, ry

    def calc_heuristic(self, n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
        return d

    def calc_position(self, index, minp):
        pos = index*self.reso+minp
        return pos

    def calc_xyindex(self, position, minp):
        return round((position - minp)/self.reso)

    def calc_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def verify_node(self, node):
        px = self.calc_position(node.x, self.minx)
        py = self.calc_position(node.y, self.miny)

        if px < self.minx:
            return False
        elif py < self.miny:
            return False
        elif px >= self.maxx:
            return False
        elif py >= self.maxy:
            return False

        if self.obmap[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.minx = round(min(ox))
        self.miny = round(min(oy))
        self.maxx = round(max(ox))
        self.maxy = round(max(oy))
        print("minx:", self.minx)
        print("miny:", self.miny)
        print("maxx:", self.maxx)
        print("maxy:", self.maxy)

        self.xwidth = round((self.maxx - self.minx)/self.reso)
        self.ywidth = round((self.maxy - self.miny)/self.reso)
        print("xwidth:", self.xwidth)
        print("ywidth:", self.ywidth)

        # obstacle map generation
        self.obmap = [[False for i in range(self.ywidth)]
                      for i in range(self.xwidth)]
        for ix in range(self.xwidth):
            x = self.calc_position(ix, self.minx)
            for iy in range(self.ywidth):
                y = self.calc_position(iy, self.miny)
                for iox, ioy in zip(ox, oy):
                    d = math.sqrt((iox - x)**2 + (ioy - y)**2)
                    if d <= self.rr:
                        self.obmap[ix][iy] = True
                        break

    def get_motion_model(self):
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = -5.0  # [m]
    sy = -5.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    dijkstra = Dijkstra(ox, oy, grid_size, robot_radius)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()


import math

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


def mod2pi(theta):
    return theta - 2.0 * math.pi * math.floor(theta / 2.0 / math.pi)


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def LSL(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d + sa - sb

    mode = ["L", "S", "L"]
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sa - sb))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((cb - ca), tmp0)
    t = mod2pi(-alpha + tmp1)
    p = math.sqrt(p_squared)
    q = mod2pi(beta - tmp1)
    #  print(np.rad2deg(t), p, np.rad2deg(q))

    return t, p, q, mode


def RSR(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d - sa + sb
    mode = ["R", "S", "R"]
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sb - sa))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((ca - cb), tmp0)
    t = mod2pi(alpha - tmp1)
    p = math.sqrt(p_squared)
    q = mod2pi(-beta + tmp1)

    return t, p, q, mode


def LSR(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = -2 + (d * d) + (2 * c_ab) + (2 * d * (sa + sb))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(-mod2pi(beta) + tmp2)

    return t, p, q, mode


def RSL(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = (d * d) - 2 + (2 * c_ab) - (2 * d * (sa + sb))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
    t = mod2pi(alpha - tmp2)
    q = mod2pi(beta - tmp2)

    return t, p, q, mode


def RLR(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["R", "L", "R"]
    tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp_rlr) > 1.0:
        return None, None, None, mode

    p = mod2pi(2 * math.pi - math.acos(tmp_rlr))
    t = mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + mod2pi(p / 2.0))
    q = mod2pi(alpha - beta - t + mod2pi(p))
    return t, p, q, mode


def LRL(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["L", "R", "L"]
    tmp_lrl = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (- sa + sb)) / 8.0
    if abs(tmp_lrl) > 1:
        return None, None, None, mode
    p = mod2pi(2 * math.pi - math.acos(tmp_lrl))
    t = mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
    q = mod2pi(mod2pi(beta) - alpha - t + mod2pi(p))

    return t, p, q, mode


def dubins_path_planning_from_origin(ex, ey, eyaw, c, D_ANGLE):
    # normalize
    dx = ex
    dy = ey
    D = math.sqrt(dx ** 2.0 + dy ** 2.0)
    d = D * c
    #  print(dx, dy, D, d)

    theta = mod2pi(math.atan2(dy, dx))
    alpha = mod2pi(- theta)
    beta = mod2pi(eyaw - theta)
    #  print(theta, alpha, beta, d)

    planners = [LSL, RSR, LSR, RSL, RLR, LRL]

    bcost = float("inf")
    bt, bp, bq, bmode = None, None, None, None

    for planner in planners:
        t, p, q, mode = planner(alpha, beta, d)
        if t is None:
            continue

        cost = (abs(t) + abs(p) + abs(q))
        if bcost > cost:
            bt, bp, bq, bmode = t, p, q, mode
            bcost = cost

    #  print(bmode)
    px, py, pyaw = generate_course([bt, bp, bq], bmode, c, D_ANGLE)

    return px, py, pyaw, bmode, bcost


def dubins_path_planning(sx, sy, syaw, ex, ey, eyaw, c, D_ANGLE=np.deg2rad(10.0)):
    """
    Dubins path plannner

    input:
        sx x position of start point [m]
        sy y position of start point [m]
        syaw yaw angle of start point [rad]
        ex x position of end point [m]
        ey y position of end point [m]
        eyaw yaw angle of end point [rad]
        c curvature [1/m]

    output:
        px
        py
        pyaw
        mode

    """

    ex = ex - sx
    ey = ey - sy

    lex = math.cos(syaw) * ex + math.sin(syaw) * ey
    ley = - math.sin(syaw) * ex + math.cos(syaw) * ey
    leyaw = eyaw - syaw

    lpx, lpy, lpyaw, mode, clen = dubins_path_planning_from_origin(
        lex, ley, leyaw, c, D_ANGLE)

    px = [math.cos(-syaw) * x + math.sin(-syaw)
          * y + sx for x, y in zip(lpx, lpy)]
    py = [- math.sin(-syaw) * x + math.cos(-syaw)
          * y + sy for x, y in zip(lpx, lpy)]
    pyaw = [pi_2_pi(iyaw + syaw) for iyaw in lpyaw]

    return px, py, pyaw, mode, clen


def generate_course(length, mode, c, D_ANGLE):

    px = [0.0]
    py = [0.0]
    pyaw = [0.0]

    for m, l in zip(mode, length):
        pd = 0.0
        if m == "S":
            d = 1.0 * c
        else:  # turning couse
            d = D_ANGLE

        while pd < abs(l - d):
            #  print(pd, l)
            px.append(px[-1] + d / c * math.cos(pyaw[-1]))
            py.append(py[-1] + d / c * math.sin(pyaw[-1]))

            if m == "L":  # left turn
                pyaw.append(pyaw[-1] + d)
            elif m == "S":  # Straight
                pyaw.append(pyaw[-1])
            elif m == "R":  # right turn
                pyaw.append(pyaw[-1] - d)
            pd += d

        d = l - pd
        px.append(px[-1] + d / c * math.cos(pyaw[-1]))
        py.append(py[-1] + d / c * math.sin(pyaw[-1]))

        if m == "L":  # left turn
            pyaw.append(pyaw[-1] + d)
        elif m == "S":  # Straight
            pyaw.append(pyaw[-1])
        elif m == "R":  # right turn
            pyaw.append(pyaw[-1] - d)
        pd += d

    return px, py, pyaw


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):  # pragma: no cover
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def main():
    print("Dubins path planner sample start!!")

    start_x = 1.0  # [m]
    start_y = 1.0  # [m]
    start_yaw = np.deg2rad(45.0)  # [rad]

    end_x = -3.0  # [m]
    end_y = -3.0  # [m]
    end_yaw = np.deg2rad(-45.0)  # [rad]

    curvature = 1.0

    px, py, pyaw, mode, clen = dubins_path_planning(start_x, start_y, start_yaw,
                                                    end_x, end_y, end_yaw, curvature)

    if show_animation:
        plt.plot(px, py, label="final course " + "".join(mode))

        # plotting
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)

        #  for (ix, iy, iyaw) in zip(px, py, pyaw):
        #  plot_arrow(ix, iy, iyaw, fc="b")

        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()


def test():

    NTEST = 5

    for i in range(NTEST):
        start_x = (np.random.rand() - 0.5) * 10.0  # [m]
        start_y = (np.random.rand() - 0.5) * 10.0  # [m]
        start_yaw = np.deg2rad((np.random.rand() - 0.5) * 180.0)  # [rad]

        end_x = (np.random.rand() - 0.5) * 10.0  # [m]
        end_y = (np.random.rand() - 0.5) * 10.0  # [m]
        end_yaw = np.deg2rad((np.random.rand() - 0.5) * 180.0)  # [rad]

        curvature = 1.0 / (np.random.rand() * 5.0)

        px, py, pyaw, mode, clen = dubins_path_planning(
            start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature)

        if show_animation:
            plt.cla()
            plt.plot(px, py, label="final course " + str(mode))

            #  plotting
            plot_arrow(start_x, start_y, start_yaw)
            plot_arrow(end_x, end_y, end_yaw)

            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.pause(1.0)

    print("Test done")


if __name__ == '__main__':
    test()
    main()


runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/test.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')
%reset
runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/test.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')
%reset
runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/test.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')

## ---(Sun Sep  8 14:27:38 2019)---
runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/test.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')
print(map_grid)
runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/test.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')

## ---(Sun Sep  8 14:34:46 2019)---
runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/test.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')
runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/2012D.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')
%reset
runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/2012D.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')
import matplotlib.pyplot as plt
plt.scatter(error)
runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/2012D.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')
print(error)
runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/2012D.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')
error=np.array([80,60],[230,60],[80,210],[230,210],[300,400],[500,600],[300,600],[500,600])
error=np.array([[80,60],[230,60],[80,210],[230,210],[300,400],[500,600],[300,600],[500,600]])
plt.scatter(error[:,0],error[:,1])
runfile('C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master/2012D.py', wdir='C:/Users/92156/Desktop/datastruct_and_algorithms-master/datastruct_and_algorithms-master')
len(X)
from sympy import Point, Circle, Line, var
import matplotlib.pyplot as plt
var('t')
c1 = Circle(Point(0, 0), 2)
c2 = Circle(Point(4, 4), 3)
l1 = Line(c1.center, c2.center)
p1 = l1.arbitrary_point(t).subs({t: -c1.radius / (c2.radius - c1.radius)})
p2 = l1.arbitrary_point(t).subs({t: c1.radius / (c1.radius + c2.radius)})
t1 = c1.tangent_lines(p1)
t2 = c1.tangent_lines(p2)
ta = t1 + t2
fig = plt.gcf()
ax = fig.gca()
ax.set_xlim((-10, 10))
ax.set_ylim((-10, 10))
ax.set_aspect(1)
cp1 = plt.Circle((c1.center.x, c1.center.y), c1.radius, fill = False)
cp2 = plt.Circle((c2.center.x, c2.center.y), c2.radius, fill = False)
tp = [0 for i in range(4)]
for i in range(4):
start = ta[i].arbitrary_point(t).subs({t:-10})
end = ta[i].arbitrary_point(t).subs({t:10})
tp[i] = plt.Line2D([start.x, end.x], [start.y, end.y], lw = 2)
ax.add_artist(cp1)
ax.add_artist(cp2)
for i in range(4):
ax.add_artist(tp[i])
from sympy import Point, Circle, Line, var
import matplotlib.pyplot as plt

var('t')

c1 = Circle(Point(0, 0), 2)
c2 = Circle(Point(4, 4), 3)
l1 = Line(c1.center, c2.center)
p1 = l1.arbitrary_point(t).subs({t: -c1.radius / (c2.radius - c1.radius)})
p2 = l1.arbitrary_point(t).subs({t:  c1.radius / (c1.radius + c2.radius)})
t1 = c1.tangent_lines(p1)
t2 = c1.tangent_lines(p2)
ta = t1 + t2

fig = plt.gcf()
ax = fig.gca()
ax.set_xlim((-10, 10))
ax.set_ylim((-10, 10))
ax.set_aspect(1)

cp1 = plt.Circle((c1.center.x, c1.center.y), c1.radius, fill = False)
cp2 = plt.Circle((c2.center.x, c2.center.y), c2.radius, fill = False)
tp = [0 for i in range(4)]
for i in range(4):
    start = ta[i].arbitrary_point(t).subs({t:-10})
    end = ta[i].arbitrary_point(t).subs({t:10})
    tp[i] = plt.Line2D([start.x, end.x], [start.y, end.y], lw = 2)

ax.add_artist(cp1)
ax.add_artist(cp2)
for i in range(4):
    ax.add_artist(tp[i])
    
%reset
from sympy import Point, Circle, Line, var
import matplotlib.pyplot as plt

var('t')

c1 = Circle(Point(150, 600), 10)
c2 = Circle(Point(220, 530), 10)
l1 = Line(c1.center, c2.center)
p1 = l1.arbitrary_point(t).subs({t: -c1.radius / (c2.radius - c1.radius)})
p2 = l1.arbitrary_point(t).subs({t:  c1.radius / (c1.radius + c2.radius)})
t1 = c1.tangent_lines(p1)
t2 = c1.tangent_lines(p2)
ta = t1 + t2

fig = plt.gcf()
ax = fig.gca()
ax.set_xlim((-10, 10))
ax.set_ylim((-10, 10))
ax.set_aspect(1)

cp1 = plt.Circle((c1.center.x, c1.center.y), c1.radius, fill = False)
cp2 = plt.Circle((c2.center.x, c2.center.y), c2.radius, fill = False)
tp = [0 for i in range(4)]
for i in range(4):
    start = ta[i].arbitrary_point(t).subs({t:-10})
    end = ta[i].arbitrary_point(t).subs({t:10})
    tp[i] = plt.Line2D([start.x, end.x], [start.y, end.y], lw = 2)

ax.add_artist(cp1)
ax.add_artist(cp2)
for i in range(4):
    ax.add_artist(tp[i])
    

## ---(Tue Sep 10 19:51:25 2019)---
runfile('C:/Users/92156/.spyder-py3/github  KNN 贝叶斯 SVM 决策树 逻辑回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svm 线性分类.py', wdir='C:/Users/92156/.spyder-py3')

## ---(Thu Sep 12 11:03:26 2019)---
#encoding=utf-8  
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

#自定义函数 e指数形式
def func(x, a,u, sig):
    return  a*(np.exp(-(x - u) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig))*(431+(4750/x))


#定义x、y散点坐标
x = [40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135]
x=np.array(x)
# x = np.array(range(20))
print('x is :\n',x)
num = [536,529,522,516,511,506,502,498,494,490,487,484,481,478,475,472,470,467,465,463]
y = np.array(num)
print('y is :\n',y)

popt, pcov = curve_fit(func, x, y,p0=[3.1,4.2,3.3])
#获取popt里面是拟合系数
a = popt[0]
u = popt[1]
sig = popt[2]


yvals = func(x,a,u,sig) #拟合y值
print(u'系数a:', a)
print(u'系数u:', u)
print(u'系数sig:', sig)

#绘图
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4) #指定legend的位置右下角
plt.title('curve_fit')
plt.show()

#encoding=utf-8  
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

#自定义函数 e指数形式
def func(x, a,u, sig):
    return  a*(np.exp(-(x - u) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig))*(431+(4750/x))


#定义x、y散点坐标
x = [40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135]
x=np.array(x)
# x = np.array(range(20))
print('x is :\n',x)
num = [536,529,522,516,511,506,502,498,494,490,487,484,481,478,475,472,470,467,465,463]
y = np.array(num)
print('y is :\n',y)

popt, pcov = curve_fit(func, x, y,p0=[3.1,4.2,3.3])
#获取popt里面是拟合系数
a = popt[0]
u = popt[1]
sig = popt[2]


yvals = func(x,a,u,sig) #拟合y值
print(u'系数a:', a)
print(u'系数u:', u)
print(u'系数sig:', sig)

#绘图
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4) #指定legend的位置右下角
plt.title('curve_fit')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
 
#自定义函数 e指数形式
def func(x, a, b,c):
    return a*np.sqrt(x)*(b*np.square(x)+c)
 
#定义x、y散点坐标
x = [20,30,40,50,60,70]
x = np.array(x)
num = [453,482,503,508,498,479]
y = np.array(num)
 
#非线性最小二乘法拟合
popt, pcov = curve_fit(func, x, y)
#获取popt里面是拟合系数
print(popt)
a = popt[0] 
b = popt[1]
c = popt[2]
yvals = func(x,a,b,c) #拟合y值
print('popt:', popt)
print('系数a:', a)
print('系数b:', b)
print('系数c:', c)
print('系数pcov:', pcov)
print('系数yvals:', yvals)
#绘图
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4) #指定legend的位置右下角
plt.title('curve_fit')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
 
#定义x、y散点坐标
x = [10,20,30,40,50,60,70,80]
x = np.array(x)
print('x is :\n',x)
num = [174,236,305,334,349,351,342,323]
y = np.array(num)
print('y is :\n',y)
#用3次多项式拟合
f1 = np.polyfit(x, y, 3)
print('f1 is :\n',f1)
 
p1 = np.poly1d(f1)
print('p1 is :\n',p1)
 
#也可使用yvals=np.polyval(f1, x)
yvals = p1(x)  #拟合y值
print('yvals is :\n',yvals)
#绘图
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4) #指定legend的位置右下角
plt.title('polyfitting')
plt.show()


## ---(Thu Sep 12 11:08:14 2019)---
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Settings
n_repeat = 50  # Number of iterations for computing expectations
n_train = 50  # Size of the training set
n_test = 1000  # Size of the test set
noise = 0.1  # Standard deviation of the noise
np.random.seed(0)

estimators = [("Tree", DecisionTreeRegressor()),
              ("RandomForestRegressor", RandomForestRegressor(random_state=100)),
              ("ExtraTreesClassifier", ExtraTreesRegressor(random_state=100)), ]

n_estimators = len(estimators)


# Generate data
def f(x):
    x = x.ravel()

    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)


def generate(n_samples, noise, n_repeat=1):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X)

    if n_repeat == 1:
        y = f(X) + np.random.normal(0.0, noise, n_samples)
    else:
        y = np.zeros((n_samples, n_repeat))

        for i in range(n_repeat):
            y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)

    X = X.reshape((n_samples, 1))

    return X, y


X_train = []
y_train = []

for i in range(n_repeat):
    X, y = generate(n_samples=n_train, noise=noise)
    X_train.append(X)
    y_train.append(y)

X_test, y_test = generate(n_samples=n_test, noise=noise, n_repeat=n_repeat)

# Loop over estimators to compare
for n, (name, estimator) in enumerate(estimators):
    # Compute predictions
    y_predict = np.zeros((n_test, n_repeat))

    for i in range(n_repeat):
        estimator.fit(X_train[i], y_train[i])
        y_predict[:, i] = estimator.predict(X_test)

    # Bias^2 + Variance + Noise decomposition of the mean squared error
    y_error = np.zeros(n_test)

    for i in range(n_repeat):
        for j in range(n_repeat):
            y_error += (y_test[:, j] - y_predict[:, i]) ** 2

    y_error /= (n_repeat * n_repeat)

    y_noise = np.var(y_test, axis=1)
    y_bias = (f(X_test) - np.mean(y_predict, axis=1)) ** 2
    y_var = np.var(y_predict, axis=1)

    print("{0}: {1:.4f} (error) = {2:.4f} (bias^2) "
          " + {3:.4f} (var) + {4:.4f} (noise)".format(name,
                                                      np.mean(y_error),
                                                      np.mean(y_bias),
                                                      np.mean(y_var),
                                                      np.mean(y_noise)))

    # Plot figures
    plt.subplot(2, n_estimators, n + 1)
    plt.plot(X_test, f(X_test), "b", label="$f(x)$")
    plt.plot(X_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")

    for i in range(n_repeat):
        if i == 0:
            plt.plot(X_test, y_predict[:, i], "r", label="$\^y(x)$")
        else:
            plt.plot(X_test, y_predict[:, i], "r", alpha=0.05)

    plt.plot(X_test, np.mean(y_predict, axis=1), "c",
             label="$\mathbb{E}_{LS} \^y(x)$")

    plt.xlim([-5, 5])
    plt.title(name)

    if n == 0:
        plt.legend(loc="upper left", prop={"size": 11})

    plt.subplot(2, n_estimators, n_estimators + n + 1)
    plt.plot(X_test, y_error, "r", label="$error(x)$")
    plt.plot(X_test, y_bias, "b", label="$bias^2(x)$"),
    plt.plot(X_test, y_var, "g", label="$variance(x)$"),
    plt.plot(X_test, y_noise, "c", label="$noise(x)$")

    plt.xlim([-5, 5])
    plt.ylim([0, 0.1])

    if n == 0:
        plt.legend(loc="upper left", prop={"size": 11})

plt.show()



import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

estimators = [("RandomForest", RandomForestClassifier(random_state=100)),
              ("ExtraTrees", ExtraTreesClassifier(random_state=100)), ]

n_estimators = len(estimators)

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)

for n, (name, estimator) in enumerate(estimators):
    estimator.fit(X, y)
    importances = estimator.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
#     print(name +" Feature ranking:")
#     for f in range(X.shape[1]):
#         print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.subplot(1, n_estimators, n + 1)
    plt.title(name + " Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
plt.show()

from sklearn import svm
>>> X = [[0, 0], [1, 1]]
>>> y = [0, 1]
>>> clf = svm.SVC()
>>> clf.fit(X, y)  
>>> clf.predict([[2., 2.]])
array([1])
>>> import xgboost as xgb

import numpy as np
from sklearn import svm
>>> X = [[0, 0], [1, 1]]
>>> y = [0, 1]
>>> clf = svm.SVC()
>>> clf.fit(X, y)  
>>> clf.predict([[2., 2.]])
array([1])
>>> import xgboost as xgb

from sklearn import svm
>>> X = [[0, 0], [1, 1]]
>>> y = [0, 1]
>>> clf = svm.SVC()
>>> clf.fit(X, y)  
>>> clf.predict([[2., 2.]])
>>> import xgboost as xgb

%Reset
%reset
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# read in the iris data
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()


dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)

# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

# 显示重要特征
plot_importance(model)
plt.show()

from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# read in the iris data
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 训练模型
model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax')
model.fit(X_train, y_train)

# 对测试集进行预测
ans = model.predict(X_test)

# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

# 显示重要特征
plot_importance(model)
plt.show()


## ---(Thu Sep 12 15:17:18 2019)---
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets 
data = datasets.load_iris().data     #数据不知道
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
knnAccuracy = metrics.accuracy_score(y_test, predicted)
knnF1score = metrics.f1_score(y_test, predicted, average='macro')
#贝叶斯
GB = GaussianNB()
GB.fit(X_train, y_train)
predicted = GB.predict(X_test)
GBAccuracy = metrics.accuracy_score(y_test, predicted)
GBF1score = metrics.f1_score(y_test, predicted, average='macro')
#决策树
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
predicted = DT.predict(X_test)
DTAccuracy = metrics.accuracy_score(y_test, predicted)
DTF1score = metrics.f1_score(y_test, predicted, average='macro')
#逻辑回归
LR = LogisticRegression()
LR.fit(X_train, y_train)
predicted = LR.predict(X_test)
LRAccuracy = metrics.accuracy_score(y_test, predicted)
LRF1score = metrics.f1_score(y_test, predicted, average='macro')
#SVM
svc = SVC()
svc.fit(X_train, y_train)
predicted = svc.predict(X_test)
svcAccuracy = metrics.accuracy_score(y_test, predicted)
svcF1score = metrics.f1_score(y_test, predicted, average='macro')

result = {"accuracy": [knnAccuracy, GBAccuracy, DTAccuracy, LRAccuracy, svcAccuracy],
        "f1-score": [knnF1score, GBF1score, DTF1score, LRF1score, svcF1score]}
results = pd.DataFrame(result, index=["knn", "贝叶斯", "决策树", "逻辑回归", "SVM"])
print("iris数据集")
print(results)
def cc(x,quantile=[0.01,0.99]):

#    """盖帽法处理异常值
#    Args：
#        x：pd.Series列，连续变量
#        quantile：指定盖帽法的上下分位数范围
#    """

# 生成分位数
    Q01,Q99=x.quantile(quantile).values.tolist()

# 替换异常值为指定的分位数
    if Q01 > x.min():
        x=x.copy()
        x.loc[x<Q01] = Q01
    if Q99 < x.max():
        x = x.copy()
        x.loc[x>Q99] = Q99
    return(x)
import seaborn as sns 
import pandas as pd 
d1=pd.read_csv("D1.csv")
d1=pd.read_csv("D1.csv",encoding="gbk")
d2=pd.read_csv("D2.csv",encoding="gbk")
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1.index=d1["时间"]
x1=d1["PM2.5"]
y1=d1["时间"]
plt.rcParams['figure.figsize'] = (12.0,5.0)
sns.relplot(x="PM2.5", y="时间", data=d1, ci=None)
runfile('C:/Users/92156/.spyder-py3/2019D.py', wdir='C:/Users/92156/.spyder-py3')
d1.describe
cc1=d1.describe()
cc1=d2.describe()
sns.boxplot(x=["PM2.5","PM10"],data=d1)
sns.boxplot(y="PM2.5",data=d1)
sns.boxplot(y="PM2.5",data=d1,notch=True)
sns.boxplot(y="PM2.5",data=d1,width=0.3)
sns.boxplot(y="PM2.5",data=d1,width=0.3,palette="Blues")
def cc(x,quantile=[0.01,0.99]):
#    """盖帽法处理异常值
#    Args：
#        x：pd.Series列，连续变量
#        quantile：指定盖帽法的上下分位数范围
#    """

# 生成分位数
    Q01,Q99=x.quantile(quantile).values.tolist()

# 替换异常值为指定的分位数
    if Q01 > x.min():
        x=x.copy()
        x.loc[x<Q01] = Q01
    if Q99 < x.max():
        x = x.copy()
        x.loc[x>Q99] = Q99
    return(x)

cc(d1["PM2.5"])
d1shan=cc(d1["PM2.5"])
cc(d1["PM2.5"],quantile=[0.1,0.9])
d1shan=cc(cc(d1["PM2.5"],quantile=[0.1,0.9]))
import numpy as np
    #new_nums = list(set(deg)) #剔除重复元素
    mean = np.mean(deg)
    var = np.var(deg)
    print("原始数据共",len(deg),"个\n",deg)
    '''
    for i in range(len(deg)):
        print(deg[i],'→',(deg[i] - mean)/var)
        #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据
    '''
    #print("中位数:",np.median(deg))
    percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
    print("分位数：",percentile)
    #以下为箱线图的五个特征值
    Q1 = percentile[0]#上四分位数
    Q3 = percentile[2]#下四分位数
    IQR = Q3 - Q1#四分位距
    ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
    llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

    new_deg = []
    for i in range(len(deg)):
        if(llim<deg[i] and deg[i]<ulim):
            new_deg.append(deg[i])
    print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        new_deg.append(deg[i])

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import pandas as pd  # 导入pandas库

# 生成异常数据
df = d1
print (df)  # 打印输出

# 通过Z-Score方法判断异常值  #阿特曼Z-score模型
df_zscore = df.copy()  # 复制一个用来存储Z-score得分的数据框
cols = df.columns  # 获得数据框的列名
for col in cols:  # 循环读取每列
    df_col = df[col]  # 得到每列的值
    z_score = (df_col - df_col.mean()) / df_col.std()  # 计算每列的Z-score得分
    df_zscore[col] = z_score.abs() > 2.2  # 判断Z-score得分是否大于2.2，如果是则是True，否则为False

print (df_zscore)  # 打印输出
%reset
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1=pd.read_csv("D1.csv",encoding="gbk")
import pandas as pd  # 导入pandas库

# 生成异常数据
df = d1
print (df)  # 打印输出

# 通过Z-Score方法判断异常值  #阿特曼Z-score模型
df_zscore = df.copy()  # 复制一个用来存储Z-score得分的数据框
cols = df.columns  # 获得数据框的列名
for col in cols:  # 循环读取每列
    df_col = df[col]  # 得到每列的值
    z_score = (df_col - df_col.mean()) / df_col.std()  # 计算每列的Z-score得分
    df_zscore[col] = z_score.abs() > 2.2  # 判断Z-score得分是否大于2.2，如果是则是True，否则为False

print (df_zscore)  # 打印输出
%reset
runfile('C:/Users/92156/.spyder-py3/2019D.py', wdir='C:/Users/92156/.spyder-py3')
print("原始数据共",len(deg),"个\n",deg)
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
sns.boxplot(y="PM2.5",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="PM10",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="CO",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="N02",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="SO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="O3",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="PM2.5",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="PM10",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="CO",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="N02",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="NO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="SO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="O3",data=d1,width=0.3,palette="Blues")
C1=sns.boxplot(y="PM2.5",data=d1,width=0.3,palette="Blues")
sns.pairplot(d1)
sns.boxplot(y="PM10",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="PM2.5",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="PM10",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="CO",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="NO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="SO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="O3",data=d1,width=0.3,palette="Blues")
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
percentile = np.percentile(deg[[""]], (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        deg[i]=False

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        deg[i]=False

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
percentile = np.percentile(deg[["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]], (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        deg[i]=False

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
percentile = np.percentile(deg[["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]], (25, 50, 75), interpolation='midpoint')
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg["PM2.5"], (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        deg[i]=False

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg["PM2.5"], (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        deg[i]=False
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg["PM2.5"], (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        deg[i]="False"

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg["PM2.5"], (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        deg[i]=["False"]

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg["PM2.5"], (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        deg[i]=666

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        deg[i]=False

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1=pd.read_csv("D1.csv",encoding="gbk")
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
new_deg = []
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
IQR
Q3
Q1
range(len(deg))
deg=list(deg)
range(len(deg))
new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        deg[i]=False
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
%reset
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1=pd.read_csv("D1.csv",encoding="gbk")
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        deg[i]=False
print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg["PM2.5"], (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        new_deg.append(deg[i])

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        new_deg.append(deg[i])

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
%reset
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1=pd.read_csv("D1.csv",encoding="gbk")
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
print("原始数据共",len(deg),"个\n",deg)
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        new_deg.append(deg[i])

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg["PM2.5"], (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

new_deg = []
for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        new_deg.append(deg[i])

print("清洗后数据共",len(new_deg),"个\n",new_deg)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        d1["PM2.5"]=False
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1=pd.read_csv("D1.csv",encoding="gbk")
d2=pd.read_csv("D2.csv",encoding="gbk")
cc1=d1.describe()
cc1=d2.describe()
sns.boxplot(y="PM2.5",data=d1,width=0.3,palette="Blues")
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

for i in range(len(deg)):
    if(llim<deg[i] and deg[i]<ulim):
        d1["PM2.5cc"]=False
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

for i in range(len(deg)):
    if deg[i]<ulim:
        d1["PM2.5cc"]=True
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1=pd.read_csv("D1.csv",encoding="gbk")
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

for i in range(len(deg)):
    if deg[i]<ulim:
        d1["PM2.5cc"]=True
deg[1]
deg[1335]<ulim
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
print("原始数据共",len(deg),"个\n",deg)
'''
for i in range(len(deg)):
    print(deg[i],'→',(deg[i] - mean)/var)
    #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据

'''
#print("中位数:",np.median(deg))
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
print("分位数：",percentile)
#以下为箱线图的五个特征值
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

for i in range(len(deg)):
    if deg[i]<ulim:
        d1["PM2.5cc"]=False
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM10"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["CO"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["NO2"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["SO2"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["O3"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["SO2"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["O3"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
from pyculiarity import detect_ts
res = detect_ts(d1, max_anoms=0.10, direction='pos',alpha=0.05) # 仅检测正向序列

from pyculiarity import detect_ts
res = detect_ts(d1, max_anoms=0.10, direction='时间',alpha=0.05) # 仅检测正向序列

import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["O3"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值

results = detect_ts(d1,
                    max_anoms=0.02,
                    direction='both', only_last='day')
                    
%reset
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1=pd.read_csv("D1.csv",encoding="gbk")
d2=pd.read_csv("D2.csv",encoding="gbk")
deg=d2["O3"]
deg[135]
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["O3"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 2*IQR#上限 非异常范围内的最大值
llim = Q1 - 2*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["SO2"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 2*IQR#上限 非异常范围内的最大值
llim = Q1 - 2*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["NO2"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 2*IQR#上限 非异常范围内的最大值
llim = Q1 - 2*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["CO"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 2*IQR#上限 非异常范围内的最大值
llim = Q1 - 2*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["PM10"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 2*IQR#上限 非异常范围内的最大值
llim = Q1 - 2*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["PM10"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["CO"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],[""],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["NO2"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["SO2"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["O3"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["SO2"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["NO2"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["O3"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["CO"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
deg=d2["PM2.5"]
mean = np.mean(deg)
var = np.var(deg)
#["PM2.5"],["PM10"],["CO"],["NO2"],["SO2"],["O3"]
percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
Q1 = percentile[0]#上四分位数
Q3 = percentile[2]#下四分位数
IQR = Q3 - Q1#四分位距
ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
print("分位数：",percentile)
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["PM2.5"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["PM10"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["CO"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["NO2"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["SO2"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["O3"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d2["PM10"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM10"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["CO"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["NO2"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["SO2"]
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
from pandas import read_csv

from pandas import datetime

from matplotlib import pyplot

from pandas.tools.plotting import autocorrelation_plot

from pandas import read_csv

from pandas import datetime

from matplotlib import pyplot

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(d1)

pyplot.show()

import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["SO2"]
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
sns.pairplot(d2)
d2=pd.read_csv("D22.csv",encoding="gbk")
cc1=d2.describe()
from itertools import groupby
data = d2["PM2.5"]

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 10):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print (k, '|', ' '.join(lst))
from itertools import groupby
data = list(d2["PM2.5"])

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 10):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print (k, '|', ' '.join(lst))
nums2=[225, 232,232,245,235,245,270,225,240,240,217,195,225,185,200,220,200,210,271,240,220,230,215,252,225,220,206,185,227,236]
for k, g in groupby(sorted(nums2), key=lambda x: int(x) // 10):
    lst = map(str, [int(y) % 10 for y in list(g)])
    print (k, '|', ' '.join(lst))
    
from itertools import groupby
data = list(d2["PM2.5"])

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 10):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print (k, '|', ' '.join(lst))
from itertools import groupby
data = list(d2["PM2.5"])

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 1):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print (k, '|', ' '.join(lst))
    
from itertools import groupby
data = '89 79 57 46 1 24 71 5 6 9 10 15 16 19 22 31 40 41 52 55 60 61 65 69 70 75 85 91 92 94'

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 10):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print k, '|', ' '.join(lst)
from itertools import groupby
data = '89 79 57 46 1 24 71 5 6 9 10 15 16 19 22 31 40 41 52 55 60 61 65 69 70 75 85 91 92 94'

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 10):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print (k, '|', ' '.join(lst))
    
from itertools import groupby
data =",".join(str(i)for i in list(d1["PM2.5"]))

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 10):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print (k, '|', ' '.join(lst))
from itertools import groupby
data =",".join(str(i)for i in list(d1["PM2.5"]))

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 5):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print (k, '|', ' '.join(lst))
from itertools import groupby
data =",".join(str(i)for i in list(d1["PM2.5"]))

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 1):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print (k, '|', ' '.join(lst))
%reset
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1=pd.read_csv("D1.csv",encoding="gbk")
d2=pd.read_csv("D22.csv",encoding="gbk")
from itertools import groupby
data =",".join(str(i)for i in list(d1["PM2.5"]))
for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 1):
    lst = map(str, [int(_) % 10 for _ in list(g)])
from itertools import groupby
data =",".join(str(i)for i in list(d1["PM2.5"]))

for k, g in groupby(sorted(data.split()), key=lambda x: int(x)):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print (k, '|', ' '.join(lst))
data[:,-1]
data[:-1]
from itertools import groupby
data = '89 79 57 46 1 24 71 5 6 9 10 15 16 19 22 31 40 41 52 55 60 61 65 69 70 75 85 91 92 94'

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 10):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print k, '|', ' '.join(lst)
from itertools import groupby
data = '89 79 57 46 1 24 71 5 6 9 10 15 16 19 22 31 40 41 52 55 60 61 65 69 70 75 85 91 92 94'

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 10):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print (k, '|', ' '.join(lst))
d2=pd.read_csv("D22.csv",encoding="gbk")
import seaborn as sns
d1=pd.read_csv("D12.csv",encoding="gbk")
cc1=d1.describe()
d1.isna()
d1.duplicated()
asdasd=d1.duplicated()
asdasd=d2.duplicated()
asdasdas=d2.isna()
d2["shifou"]=d2.duplicated()
len(d2["shifou"]==True)
d2["shifou"]==True
sum(d2["shifou"]==True)
d2.to_csv("cc1.csv")
d2 = d2[np.isnan(d2['shifou']) == False]
d2 = d2['shifou']) == False
d2 = d2['shifou'] == False
d2=pd.read_csv("D22.csv",encoding="gbk")
d2["shifou"]=d2.duplicated()
d2=d2[d2['shifou'].isin([True])]
d2=pd.read_csv("D22.csv",encoding="gbk")
d2["shifou"]=d2.duplicated()
d2=d2[d2['shifou'].isin([False])]
d2["shifou"]=d2.duplicated()
d2=d2[d2['shifou'].isin([False])]
d2.to_csv("D23.csv")
%reset
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1=pd.read_csv("D12.csv",encoding="gbk")
d2=pd.read_csv("D22.csv",encoding="gbk")
cc1=d1.describe()
cc1=d2.describe()
sns.boxplot(y="PM10",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="CO",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="NO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="SO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="O3",data=d1,width=0.3,palette="Blues")
deg=d1["SO2"]
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
d1=pd.read_csv("D1.csv",encoding="gbk")
deg=d1["SO2"]
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
# 自相关系数
import numpy as np
x=np.array(d2["PM2.5"])
def autocorrelation(x,lags):
# 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
# lags表示错开的时间间隔
  n = len(x)
  x = np.array(x)
  result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    /(x[i:].std()*x[:n-i].std()*(n-i)) \
    for i in range(1,lags+1)]
  return result

z = autocorrelation(x, 1)
print(z)
import matplotlib.pyplot as plt
import numpy as np

def autocorrelation(x,lags):
# 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
# lags表示错开的时间间隔
# Notice: 标题神马的不支持中文 #
  n = len(x)
  x = np.array(x)
  result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    /(x[i:].std()*x[:n-i].std()*(n-i)) \
    for i in range(1,lags+1)]
  return result


fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

# main figure
x1 = np.linspace(1, len(df.ix[:, 1]), len(df.ix[:, 1]))
axes1.plot(x1, df.ix[:, 1], 'r')
axes1.set_xlabel(df.columns[1])
axes1.set_ylabel('value')
axes1.set_title('main')

# insert
x2 = np.linspace(1, 10, 10)
y2 = autocorrelation(df.ix[:, 1], 10)
y2 = np.array(y2)
axes2.plot(x2, y2, 'g')
axes2.set_xlabel('jieci')
axes2.set_ylabel('ar')
axes2.set_title('autoRelation of different jieci')
plt.show()

# 自相关系数
import numpy as np
x=np.array(d2["PM2.5"])
def autocorrelation(x,lags):
# 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
# lags表示错开的时间间隔
  n = len(x)
  x = np.array(x)
  result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    /(x[i:].std()*x[:n-i].std()*(n-i)) \
    for i in range(1,lags+1)]
  return result

z = autocorrelation(x, 1)
print(z)
import matplotlib.pyplot as plt
import numpy as np

def autocorrelation(x,lags):
# 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
# lags表示错开的时间间隔
# Notice: 标题神马的不支持中文 #
  n = len(x)
  x = np.array(x)
  result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    /(x[i:].std()*x[:n-i].std()*(n-i)) \
    for i in range(1,lags+1)]
  return result


fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

# main figure
x1 = np.linspace(1, len(x), len(x))
axes1.plot(x1, df.ix[:, 1], 'r')
axes1.set_xlabel(df.columns[1])
axes1.set_ylabel('value')
axes1.set_title('main')

# insert
x2 = np.linspace(1, 10, 10)
y2 = autocorrelation(df.ix[:, 1], 10)
y2 = np.array(y2)
axes2.plot(x2, y2, 'g')
axes2.set_xlabel('jieci')
axes2.set_ylabel('ar')
axes2.set_title('autoRelation of different jieci')
plt.show()

# 自相关系数
import numpy as np
x=np.array(d2["PM2.5"])
def autocorrelation(x,lags):
# 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
# lags表示错开的时间间隔
  n = len(x)
  x = np.array(x)
  result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    /(x[i:].std()*x[:n-i].std()*(n-i)) \
    for i in range(1,lags+1)]
  return result

z = autocorrelation(x, 1)
print(z)
import matplotlib.pyplot as plt
import numpy as np

def autocorrelation(x,lags):
# 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
# lags表示错开的时间间隔
# Notice: 标题神马的不支持中文 #
  n = len(x)
  x = np.array(x)
  result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    /(x[i:].std()*x[:n-i].std()*(n-i)) \
    for i in range(1,lags+1)]
  return result


fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

# main figure
x1 = np.linspace(1, len(x), len(x))
axes1.plot(x1, x, 'r')
axes1.set_xlabel(df.columns[1])
axes1.set_ylabel('value')
axes1.set_title('main')

# insert
x2 = np.linspace(1, 10, 10)
y2 = autocorrelation(x, 10)
y2 = np.array(y2)
axes2.plot(x2, y2, 'g')
axes2.set_xlabel('jieci')
axes2.set_ylabel('ar')
axes2.set_title('autoRelation of different jieci')
plt.show()

import numpy as np
x=np.array(d2["PM2.5"])
def autocorrelation(x,lags):

# 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
# lags表示错开的时间间隔
  n = len(x)
  x = np.array(x)
  result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    /(x[i:].std()*x[:n-i].std()*(n-i)) \
    for i in range(1,lags+1)]
  return result

z = autocorrelation(x, 1)
print(z)
import numpy as np
x=np.array(d2["PM2.5"])
def autocorrelation(x,lags):

# 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
# lags表示错开的时间间隔
  n = len(x)
  x = np.array(x)
  result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    /(x[i:].std()*x[:n-i].std()*(n-i)) \
    for i in range(1,lags+1)]
  return result

z = autocorrelation(x, 600)
print(z)
import numpy as np
x=np.array(d2["PM2.5"])
def autocorrelation(x,lags):

# 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
# lags表示错开的时间间隔
  n = len(x)
  x = np.array(x)
  result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    /(x[i:].std()*x[:n-i].std()*(n-i)) \
    for i in range(1,lags+1)]
  return result

z = autocorrelation(x, 600)
print(z)
plt.plot(z)
import numpy as np
x=np.array(d2["PM10"])
def autocorrelation(x,lags):

# 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
# lags表示错开的时间间隔
  n = len(x)
  x = np.array(x)
  result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    /(x[i:].std()*x[:n-i].std()*(n-i)) \
    for i in range(1,lags+1)]
  return result

z = autocorrelation(x, 600)
print(z)
plt.plot(z)
import numpy as np
x=np.array(d2["PM10"])
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
from scipy import stats
import numpy as np
x = np.arange(d2["PM2.5"])
y = stats.norm.cdf(x, 0, 1)
plt.plot(x, y)
from scipy import stats
import numpy as np
x = np.arange(d2["PM2.5"].all())
y = stats.norm.cdf(x, 0, 1)
plt.plot(x, y)
from scipy import stats
import numpy as np
x = np.arange(d2["PM2.5"].all())
y = stats.norm.cdf(x)
plt.plot(x, y)
d2["PM2.5"]
np.array(d2["PM10"])
from scipy import stats
import numpy as np
x = np.array(d2["PM2.5"])
y = stats.norm.cdf(x, 0, 1)
plt.plot(x, y)
import scipy
_,pvalue= scipy.stats.jarque_bera(x)
x=np.array(d2["CO"])
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
x=np.array(d2["SO2"])
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
%reset
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1=pd.read_csv("D1.csv",encoding="gbk")
d2=pd.read_csv("D2.csv",encoding="gbk")
cc1=d1.describe()
cc1=d2.describe()
d1=pd.read_csv("D12.csv",encoding="gbk")
d2=pd.read_csv("D22.csv",encoding="gbk")
d2["shifou"]=d2.duplicated()
d2=d2[d2['shifou'].isin([False])]
d2.to_csv("D23.csv")
import pandas as pd
d1=pd.read_csv("D24零点漂移",encoding="gbk")
d1=pd.read_csv("D24零点漂移",encoding="gbk")
import pandas as pd
d1=pd.read_csv("D24零点漂移",encoding="gbk")
d1=pd.read_csv("D24零点漂移")
d1=pd.read_csv("D1.csv",encoding="gbk")
d1=pd.read_csv("D240dian.csv")
c1=d1.drop_duplicates()
c1=d1.drop_duplicates('xiaoshi')
c1[:,1]-c1[0]
c1.iloc[:,0]-c1.iloc[:,1]
c1.iloc[0,:]-c1.iloc[1,:]
c1.to_csv("D240dian.csv")
c1= c1.drop('xiaoshi', 1)
c1= c1.drop('shifou', 1)
c1= c1.drop('shijian', 1)
c1.iloc[0,:]-c1.iloc[1,:]
range(len(c1))
for i in range(len(c1)):
    print(i)
s=[]
for i in range(len(c1)):
    s.append(c1.iloc[i,:]-c1.iloc[i+1,:])
s=[]
for i in range(len(c1)):
    s.append(c1.iloc[i+1,:]-c1.iloc[i,:])
s = pd.DataFrame(s)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[i,:])/20000)

s=pd.DataFrame(s)
s=pd.DataFrame(s)
s.to_csv("Dlingdian.csv")
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/20000)
s=pd.DataFrame(s)
s.to_csv("Dlingdian0.csv")
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/0.5)

s=pd.DataFrame(s)
aa=max(s)
s=pd.DataFrame(s)
aa=max(s)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/50)
s=pd.DataFrame(s)
aa=max(s)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[i,:])/50)
s=pd.DataFrame(s)
aa=max(s)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[i,:])/2000)
s=pd.DataFrame(s)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[i,:])/500)
s=pd.DataFrame(s)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/500)
s=pd.DataFrame(s)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/200000)
s=pd.DataFrame(s)
%reset
import pandas as pd 
d1=pd.read_csv("D23shanchong.csv")
d1=pd.read_csv("D23shanchong.csv",encoding="gbk")
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1=pd.read_csv("D1.csv",encoding="gbk")
import seaborn as sns
sns.boxplot(y="PM10",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="PM10",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="CO",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="NO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="SO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="O3",data=d1,width=0.3,palette="Blues")
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
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
deg=d1["SO2"]
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
deg=d1["NO2"]
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
d1=pd.read_csv("D1.csv",encoding="gbk")
cc1=d1.describe()
d1=pd.read_csv("D1.csv",encoding="gbk")
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["SO2"]
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
d1=pd.read_csv("D1.csv",encoding="gbk")
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["SO2"]
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
sns.boxplot(y="SO2",data=d1,width=0.3,palette="Blues")
d1=pd.read_csv("D240dian.csv")
%reset
d1=pd.read_csv("D240dian.csv")
import pandas as pd
d1=pd.read_csv("D240dian.csv")
for i in d1.groupby(["xiaoshi"]):
    print(i)
    
data =d1.groupby("xiaoshi").mean()
d1=pd.read_csv("D240dian.csv")
data =d1.groupby("xiaoshi").mean()
data.to_csv("qujunzhi.csv")
A=data.iloc[2,:].mean()
A=data.iloc[2,:0].mean()
A=data.iloc[2,0].mean()
A=data.iloc[0:2,0].mean()
d1=pd.read_csv("D240dian.csv")
import pandas as pd
d1=pd.read_csv("D240dian.csv")
data =d1.groupby("xiaoshi").mean()
s=[]
A=data.iloc[0:2,0].mean()
for i in range(len(data)):
    s.append((c1.iloc[i+3,:]-A)/500)
import pandas as pd
d1=pd.read_csv("D240dian.csv")
data =d1.groupby("xiaoshi").mean()
s=[]
A=data.iloc[0:2,0].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/500)
s=pd.DataFrame(s)
s=pd.DataFrame(s)
s.to_csv("du.csv")
data =d1.groupby("xiaoshi").mean()
s=[]
A=data.iloc[0:2,1].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/2000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
A=data.iloc[0:2,2].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/2000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
A=data.iloc[0:2,2].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/20000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
A=data.iloc[0:2,3].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/200000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
A=data.iloc[0:2,4].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/200000)
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/50)
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
A=data.iloc[0:2,5].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/20000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
import pandas as pd 
d1=pd.read_csv("D24shanpiaoyihou.csv",encoding="gbk")
%reset
import pandas as pd 
d1=pd.read_csv("D24shanpiaoyihou.csv",encoding="gbk")
import pandas as pd 
d1=pd.read_csv("D24shanpiaoyihou.csv",encoding="gbk")
d2=pd.read_csv("D12.csv",encoding="gbk")
data =d1.groupby("xiaoshi").mean()
d1=pd.read_csv("D24shanpiaoyihou.csv",encoding="gbk")
data =d1.groupby("xiaoshi").mean()
d1=pd.read_csv("D24shanpiaoyihou.csv",encoding="gbk")
data =d1.groupby("xiaoshi").mean()
d1=pd.read_csv("D24shanpiaoyihou.csv",encoding="gbk")
data =d1.groupby("xiaoshi").mean()
d2=pd.read_csv("D12.csv",encoding="gbk")
d2=pd.read_csv("D12.csv",encoding="gbk")
data1=pd.merge(data,d2,on='xiaoshi')
data=drop("index")
data=data.drop("index")
d1.isna()
data=data.drop("index",1)
data1=data1.drop("index",1)
data =d1.groupby("xiaoshi").mean()
data1=pd.merge(data,d2,on='xiaoshi')
data1.to_csv("pipeihou.csv")
d1=pd.read_csv("D24shanpiaoyihou.csv",encoding="gbk")
data =d1.groupby("xiaoshi").mean()
d2=pd.read_csv("D12.csv",encoding="gbk")
data1=pd.merge(data,d2,on='xiaoshi')
data1=data1.drop("index",1)
data1.to_csv("pipeihou.csv")
data1["cha"]=data1["PM2.5_x"]-data1["PM2.5_y"]
X =np.array(data1[["PM10_x"],["CO_x"]]).reshape(-1,1)
import numpy as np
X =np.array(data1[["PM10_x"],["CO_x"]]).reshape(-1,1)
X =np.array(data1[["PM10_x"]]).reshape(-1,1)
y=np.array(data1[["cha"]]).reshape(-1,1)
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
polynomial_svm_clf = Pipeline([ ("poly_featutres", PolynomialFeatures(degree=3)),
                                ("scaler", StandardScaler()),
                                ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42)  )
                            ])
polynomial_svm_clf.fit( X_train, Y_train )

result =polynomial_svm_clf(X_train)  # 使用模型预测值
print('预测结果：',result)  # 输出预测值[-1. -1.  1.  1.]


clf = SVR(kernel='rbf', class_weight='balanced',)
clf.fit(X_train, Y_train)
y_predict = clf.predict(X_test)
#error = 0
#for i in range(len(X_test)):
#    if clf.predict([X_test[i]])[0] != Y_test[i]:
#        error +=1
#print( 'SVM错误率: %.4f' % (error/float(len(X_test))))
print( 'SVM精确率: ', precision_score(Y_test, y_predict, average='macro'))
print( 'SVM召回率: ', recall_score(Y_test, y_predict, average='macro'))
print( 'F1: ', f1_score(Y_test, y_predict, average='macro'))



from sklearn.neighbors import KNeighborsClassifier as KNN
knc = KNN(n_neighbors =6,)
knc.fit(X_train,Y_train)
y_predict = knc.predict(X_test)
print('KNN准确率',knc.score(X_test,Y_test))
print('KNN精确率',precision_score(Y_test, y_predict,  average='macro'))
print('KNN召回率',recall_score(Y_test, y_predict,  average='macro'))
print('F1',f1_score(Y_test, y_predict,  average='macro'))


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
y_predict = rfc.predict(X_test)
print('随机森林准确率',rfc.score(X_test, Y_test))
print('随机森林精确率',precision_score(Y_test, y_predict,  average='macro'))
print('随机森林召回率',recall_score(Y_test, y_predict,  average='macro'))
print('F1',f1_score(Y_test, y_predict,  average='macro'))
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.23)


# contour函数是画出轮廓，需要给出X和Y的网格，以及对应的Z，它会画出Z的边界（相当于边缘检测及可视化）


polynomial_svm_clf = Pipeline([ ("poly_featutres", PolynomialFeatures(degree=3)),
                                ("scaler", StandardScaler()),
                                ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42)  )
                            ])
polynomial_svm_clf.fit( X_train, Y_train )

result =polynomial_svm_clf(X_train)  # 使用模型预测值
print('预测结果：',result)  # 输出预测值[-1. -1.  1.  1.]


clf = SVR(kernel='rbf', class_weight='balanced',)
clf.fit(X_train, Y_train)
y_predict = clf.predict(X_test)
#error = 0
#for i in range(len(X_test)):
#    if clf.predict([X_test[i]])[0] != Y_test[i]:
#        error +=1
#print( 'SVM错误率: %.4f' % (error/float(len(X_test))))
print( 'SVM精确率: ', precision_score(Y_test, y_predict, average='macro'))
print( 'SVM召回率: ', recall_score(Y_test, y_predict, average='macro'))
print( 'F1: ', f1_score(Y_test, y_predict, average='macro'))



from sklearn.neighbors import KNeighborsClassifier as KNN
knc = KNN(n_neighbors =6,)
knc.fit(X_train,Y_train)
y_predict = knc.predict(X_test)
print('KNN准确率',knc.score(X_test,Y_test))
print('KNN精确率',precision_score(Y_test, y_predict,  average='macro'))
print('KNN召回率',recall_score(Y_test, y_predict,  average='macro'))
print('F1',f1_score(Y_test, y_predict,  average='macro'))


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
y_predict = rfc.predict(X_test)
print('随机森林准确率',rfc.score(X_test, Y_test))
print('随机森林精确率',precision_score(Y_test, y_predict,  average='macro'))
print('随机森林召回率',recall_score(Y_test, y_predict,  average='macro'))
print('F1',f1_score(Y_test, y_predict,  average='macro'))
model=LinearSVC(C=10, loss="hinge", random_state=42)
model.fit(X_train,Y_train)
import numpy as np  # numpy库
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库
from sklearn.model_selection import train_test_split

# 数据准备

#cc1=pd.read_excel("rizhenduan1.xls")

X =np.array(data1[["PM10_x"]]).reshape(-1,1)
y=np.array(data1[["cha"]]).reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.23)

# 训练回归模型
n_folds = 6  # 设置交叉检验的次数
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
    scores = cross_val_score(model, X, y, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
    cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
    pre_y_list.append(model.fit(X, y).predict(X))  # 将回归训练中得到的预测y存入列表

# 模型效果指标评估
n_samples, n_features = X.shape  # 总样本量,总特征数
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(5):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(y, pre_y_list[i])  # 计算每个回归指标结果
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
plt.plot(np.arange(X.shape[0]), y, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(X.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线

plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
runfile('C:/Users/92156/.spyder-py3/GDBT.py', wdir='C:/Users/92156/.spyder-py3')
%reset
import pandas as pd 
from sklearn.datasets import load_iris
data=load_iris()
%reset
import pandas as pd 
d1=pd.read_csv("D24shanpiaoyihou.csv",encoding="gbk")
d1=pd.read_csv("D24shanpiaoyihou.csv",encoding="gbk")
cm=d1.corr()
d1=d1.drop("index",axis=1)
cm=d1.corr()
import seaborn as sns
sns.heatmap(cm)
sns.heatmap(cm,annot=True, vmax=1, square=True, cmap="Blues")
sns.heatmap(cm,annot=True, vmax=0.001, square=True, cmap="Blues")
sns.heatmap(cm, vmax=0.001, square=True, cmap="Blues")
sns.heatmap(cm, vmax=0.001, cmap="Blues")
sns.pairplot(d1)
import pandas as pd
d1=pd.read_csv("D26chazhiso2.csv",encoding="gbk")
import seaborn as sns 
sns.pairplot(d1)
import pandas as pd
d1=pd.read_csv("D240dian.csv")
c1=d1.drop_duplicates('xiaoshi')
%reset
import pandas as pd
d1=pd.read_csv("D240dian.csv")
data =d1.groupby("xiaoshi").mean()
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/500)

s=pd.DataFrame(s)
#量程漂移
c1=d1.drop_duplicates('xiaoshi')
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/500)

s=pd.DataFrame(s)
import pandas as pd
d1=pd.read_csv("D240dian.csv")
c1=d1.drop_duplicates('xiaoshi')
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/500)
c1=c1.drop("shijian",axis=1)
c1=c1.drop("xiaoshi",axis=1)
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/500)

s=pd.DataFrame(s)
s=pd.DataFrame(s)
s.to_csv("du.csv")
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:1]-c1.iloc[0,:1])/500)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:1]-c1.iloc[0,:1])/500)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:1]-c1.iloc[0,:1])/2000)
s=pd.DataFrame(s)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:1]-c1.iloc[0,:1])/2000)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:2]-c1.iloc[0,:2])/2000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
c1=c1.drop("xiaoshi",axis=1)
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/2000)
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/2000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/20000)

s=pd.DataFrame(s)
s.to_csv("du.csv")
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/200000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/200000)
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/50)
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
for i in range(len(c1)):
    s.append((c1.iloc[i+1,:]-c1.iloc[0,:])/20000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
%reset
import pandas as pd
d1=pd.read_csv("D240dian.csv")
data =d1.groupby("xiaoshi").mean()
s=[]
import pandas as pd
d1=pd.read_csv("D240dian.csv")
data =d1.groupby("xiaoshi").mean()
s=[]
A=data.iloc[0:2,1].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/500)
s=pd.DataFrame(s)
s.to_csv("du.csv")
import pandas as pd
d1=pd.read_csv("D240dian.csv")
data =d1.groupby("xiaoshi").mean()
s=[]
A=data.iloc[0:2,0].mean()
A=data.iloc[0:2,0].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/500)
s=pd.DataFrame(s)
s.to_csv("du.csv")
import pandas as pd
d1=pd.read_csv("D240dian.csv")
data =d1.groupby("xiaoshi").mean()
s=[]
A=data.iloc[0:2,1].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/2000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
A=data.iloc[0:2,2].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/20000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
A=data.iloc[0:2,3].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/200000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
A=data.iloc[0:2,4].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/50)
s=pd.DataFrame(s)
s.to_csv("du.csv")
s=[]
A=data.iloc[0:2,5].mean()
for i in range(len(data)):
    s.append((data.iloc[i+3,:]-A)/20000)
s=pd.DataFrame(s)
s.to_csv("du.csv")
import pandas as pd 
d1=pd.read_csv("tongyihoupiaoyi.csv",encoding="gbk")
import pandas as pd 
d1=pd.read_excel("tongyihoupiaoyi.xls",encoding="gbk")
data =d1.groupby("xiaoshi").mean()
d2=pd.read_csv("D12.csv",encoding="gbk")
data1=pd.merge(data,d2,on='xiaoshi')
data1=pd.merge(d1,d2,on='xiaoshi')
d1=pd.read_excel("tongyihoupiaoyi.xls",encoding="gbk")
data =d1.groupby("xiaoshi").mean()
data1=pd.merge(data,d2,on='xiaoshi')
data1.to_csv("pipeihoupiaoyi.csv")
%reset
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
d1=pd.read_csv("D3shanchong.csv",encoding="gbk")
d1=pd.read_csv("D23shanchong.csv",encoding="gbk")
sns.boxplot(y="PM10",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="CO",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="NO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="SO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="O3",data=d1,width=0.3,palette="Blues")
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["SO2"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM10"]
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
deg=d1["PM10"]
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
deg=d1["CO"]
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
sns.boxplot(y="PM2.5",data=d1,width=0.3,palette="Blues")
deg=d1["NO2"]
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
d1=pd.read_csv("D12.csv",encoding="gbk")
%reset
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
d1=pd.read_csv("D12.csv",encoding="gbk")
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["NO2"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["SO2"]
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
d1=pd.read_csv("D23shanchong.csv",encoding="gbk")
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["SO2"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["NO2"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["NO2"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["CO"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM10"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
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
d1=pd.read_csv("D12.csv",encoding="gbk")
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM2.5"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["PM10"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["CO"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["NO2"]
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
import numpy as np
#new_nums = list(set(deg)) #剔除重复元素
deg=d1["SO2"]
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

from sklearn import datasets
boston = datasets.load_boston()
x,y = boston.data,boston.target

%reset

from sklearn import datasets
boston = datasets.load_boston()
x,y = boston.data,boston.target

runfile('C:/Users/92156/.spyder-py3/神经网络 回归.py', wdir='C:/Users/92156/.spyder-py3')
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
import pandas as pd
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhio3"])
x=np.array(d1[["fensu"],["yaqian"]])
x=np.array(d1[["fensu","yaqian"]])
x=np.array(d1[["fensu","yaqiang"]])
runfile('C:/Users/92156/.spyder-py3/神经网络 回归.py', wdir='C:/Users/92156/.spyder-py3')
import json
import random
import sys
import numpy as np

#### Define the quadratic and cross-entropy cost functions
class CrossEntropyCost(object):


#  @staticmethod
  def fn(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

#  @staticmethod
  def delta(z, a, y):
    return (a-y)

#### Main Network class
class Network(object):
  
  def __init__(self, sizes, cost=CrossEntropyCost):
    
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.default_weight_initializer()
    self.cost=cost
  
  def default_weight_initializer(self):
    
    self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
    self.weights = [np.random.randn(y, x)/np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])]
  def large_weight_initializer(self):
    
    self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
    self.weights = [np.random.randn(y, x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])]
  def feedforward(self, a):
    """Return the output of the network if ``a`` is input."""
    for b, w in zip(self.biases[:-1], self.weights[:-1]): # 前n-1层
      a = sigmoid(np.dot(w, a)+b)
    
    b = self.biases[-1]  # 最后一层
    w = self.weights[-1]
    a = np.dot(w, a)+b
    return a
  
  def SGD(self, training_data, epochs, mini_batch_size, eta,
      lmbda = 0.0,
      evaluation_data=None,
      monitor_evaluation_accuracy=False): # 用随机梯度下降算法进行训练
    
    n = len(training_data)
    
    for j in xrange(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
      
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
      print ("Epoch %s training complete" % j)
      
      if monitor_evaluation_accuracy:
        print ("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), j))
  
  def update_mini_batch(self, mini_batch, eta, lmbda, n):
    """Update the network's weights and biases by applying gradient
    descent using backpropagation to a single mini batch. The
    ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
    learning rate, ``lmbda`` is the regularization parameter, and
    ``n`` is the total size of the training data set.
    """
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
            for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb
            for b, nb in zip(self.biases, nabla_b)]
  
  def backprop(self, x, y):
    """Return a tuple ``(nabla_b, nabla_w)`` representing the
    gradient for the cost function C_x. ``nabla_b`` and
    ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    to ``self.biases`` and ``self.weights``."""
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(self.biases[:-1], self.weights[:-1]):  # 正向传播 前n-1层
      
      z = np.dot(w, activation)+b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)

# 最后一层，不用非线性
    b = self.biases[-1]
    w = self.weights[-1]
    z = np.dot(w, activation)+b
    zs.append(z)
    activation = z
    activations.append(activation)
    # backward pass 反向传播
    delta = (self.cost).delta(zs[-1], activations[-1], y)  # 误差 Tj - Oj 
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # (Tj - Oj) * O(j-1)
    
    for l in xrange(2, self.num_layers):
      z = zs[-l]  # w*a + b
      sp = sigmoid_prime(z) # z * (1-z)
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # z*(1-z)*(Err*w) 隐藏层误差
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) # Errj * Oi
    return (nabla_b, nabla_w)
  
  def accuracy(self, data):
    
    results = [(self.feedforward(x), y) for (x, y) in data] 
    alist=[np.sqrt((x[0][0]-y[0])**2+(x[1][0]-y[1])**2) for (x,y) in results]
    
    return np.mean(alist)
  
  def save(self, filename):
    """Save the neural network to the file ``filename``."""
    data = {"sizes": self.sizes,
        "weights": [w.tolist() for w in self.weights],
        "biases": [b.tolist() for b in self.biases],
        "cost": str(self.cost.__name__)}
    f = open(filename, "w")
    json.dump(data, f)
    f.close()

#### Loading a Network
def load(filename):
  """Load a neural network from the file ``filename``. Returns an
  instance of Network.
  """
  f = open(filename, "r")
  data = json.load(f)
  f.close()
  cost = getattr(sys.modules[__name__], data["cost"])
  net = Network(data["sizes"], cost=cost)
  net.weights = [np.array(w) for w in data["weights"]]
  net.biases = [np.array(b) for b in data["biases"]]
  return net


def sigmoid(z):
  """The sigmoid function."""
  return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
  """Derivative of the sigmoid function."""
  return sigmoid(z)*(1-sigmoid(z))
import my_datas_loader_1
import network_0

training_data,test_data = my_datas_loader_1.load_data_wrapper()
#### 训练网络，保存训练好的参数
net = network_0.Network([14,100,2],cost = network_0.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data,1000,316,0.005,lmbda =0.1,evaluation_data=test_data,monitor_evaluation_accuracy=True)
filename=r'C:\Users\hyl\Desktop\Second_158\Regression_Model\parameters.txt'
net.save(filename)
for i in range(1,11):
    ANNmodel = MLPClassifier(
            activation='relu',   #激活函数为relu,类似于s型函数
           hidden_layer_sizes=i)  #隐藏层为i
    ANNmodel.fit(inputData,outputData)  #训练模型
    score = ANNmodel.score(inputData,outputData)  #模型评分
    print(str(i) + ',' + str(score))  #每次循环都打印模型评分
from sklearn.neural_network import MLPClassifier

for i in range(1,11):
    ANNmodel = MLPClassifier(
            activation='relu',   #激活函数为relu,类似于s型函数
           hidden_layer_sizes=i)  #隐藏层为i
    ANNmodel.fit(inputData,outputData)  #训练模型
    score = ANNmodel.score(inputData,outputData)  #模型评分
    print(str(i) + ',' + str(score))  #每次循环都打印模型评分
from sklearn.neural_network import MLPClassifier

for i in range(1,11):
    ANNmodel = MLPClassifier(
            activation='relu',   #激活函数为relu,类似于s型函数
           hidden_layer_sizes=i)  #隐藏层为i
    ANNmodel.fit(x,y)  #训练模型
    score = ANNmodel.score(inputData,outputData)  #模型评分
    print(str(i) + ',' + str(score))  #每次循环都打印模型评分
x=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","shijian","PM10_x","CO_x","NO2_x","SO2_x","O3_x"]])
from sklearn.neural_network import MLPClassifier

for i in range(1,11):
    ANNmodel = MLPClassifier(
            activation='relu',   #激活函数为relu,类似于s型函数
           hidden_layer_sizes=i)  #隐藏层为i
    ANNmodel.fit(x,y)  #训练模型
    score = ANNmodel.score(inputData,outputData)  #模型评分
    print(str(i) + ',' + str(score))  #每次循环都打印模型评分
x=pd.DataFrame(x)
y=pd.DataFrame(y)
from sklearn.neural_network import MLPClassifier

for i in range(1,11):
    ANNmodel = MLPClassifier(
            activation='relu',   #激活函数为relu,类似于s型函数
           hidden_layer_sizes=i)  #隐藏层为i
    ANNmodel.fit(x,y)  #训练模型
    score = ANNmodel.score(inputData,outputData)  #模型评分
    print(str(i) + ',' + str(score))  #每次循环都打印模型评分
y=np.array(d1["chazhio3"])
x=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","PM10_x","CO_x","NO2_x","SO2_x","O3_x"]])
from sklearn.neural_network import MLPClassifier

for i in range(1,11):
    ANNmodel = MLPClassifier(
            activation='relu',   #激活函数为relu,类似于s型函数
           hidden_layer_sizes=i)  #隐藏层为i
    ANNmodel.fit(x,y)  #训练模型
    score = ANNmodel.score(inputData,outputData)  #模型评分
    print(str(i) + ',' + str(score))  #每次循环都打印模型评分
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
rbf_svr.intercept_
linear_svr.intercept_
linear_svr.coef_
import pandas as pd 

from sklearn.cross_validation import train_test_split

import numpy as np;
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
%reset
from sklearn.datasets import load_boston

import pandas as pd 

from sklearn.cross_validation import train_test_split

import numpy as np;
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhiso2"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np;
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhiso2"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)
ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
X_train = ss_X.fit_transform(X_train)
y_train = ss_y.fit_transform(y_train)
X_test = ss_X.transform(X_test)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR
linear_svr = SVR(kernel = 'linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)
poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
%reset
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
model_gbr.criterion
rbf_svr.intercept_
model_gbr.coef_
runfile('C:/Users/92156/.spyder-py3/GBT波士顿.py', wdir='C:/Users/92156/.spyder-py3')
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
 ###############################################################################
 # Load data
boston = datasets.load_boston()
import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
 ###############################################################################
 # Load data
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
%reset
import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
 ###############################################################################
 # Load data
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
  'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
model_lr.intercept_
model_gbr
model_gbr.criterion
model_gbr.estimators_
model_gbr.feature_importances_
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
model_gbr.feature_importances_
%reset
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
model_gbr.get_params
model_gbr.init
model_gbr.init_
model_gbr.learning_rate
model_gbr.n_features
model_gbr.score
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
rbf_svr.intercept_
import pandas as pd
d1=pd.read_excel("PM2.5yuPM10.xlsx",encoding="gbk")
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
d1=pd.read_excel("PM2.5yuPM10.xlsx",encoding="gbk")
Q=d1["PM2.5_y"]
B=d1["预测OM2.5"]
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(5):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(Q, B)  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
d1=pd.read_excel("PM2.5yuPM10.xlsx",encoding="gbk")
Q=d1["PM2.5_y"]
B=d1["预测PM2.5"]
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(5):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(Q, B)  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
d1=pd.read_excel("PM2.5yuPM10.xlsx",encoding="gbk")
Q=d1["PM2.5_y"]
B=d1["预测PM2.5"]
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(5):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(Q, B)  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print (70 * '-')  # 打印分隔线
print ('regression metrics:')  # 打印输出标题
print (df2)  # 打印输出回归指标的数据框
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
d1=pd.read_excel("PM2.5yuPM10.xlsx",encoding="gbk")
Q=d1["PM2.5_y"]
B=d1["预测PM2.5"]
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(5):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(Q, B)  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

df2 = pd.DataFrame(model_metrics_list, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print (70 * '-')  # 打印分隔线
print ('regression metrics:')  # 打印输出标题
print (df2)  # 打印输出回归指标的数据框
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
print(pre_y_list)
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
%reset
%reset
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
clc
clear
rbf_svr.intercept_
rbf_svr.coef_
rbf_svr.coef0
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
rbf_svr.coef_
linear_svr.coef_
rbf_svr.coef_
model_lr.intercept_
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
model_gbr.init
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
rbf_svr.intercept_
rbf_svr.coef_
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
rbf_svr.intercept_
rbf_svr.coef_
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
rbf_svr.coef_
rbf_svr.intercept_
rbf_svr.cache_size
rbf_svr.degree
rbf_svr.kernel
rbf_svr.support_
rbf_svr.dual_coef_
rbf_svr.epsilon
linear_svr.coef_
poly_svr.intercept_
poly_svr.coef_
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
rbf_svr.corf_
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
rbf_svr.corf_
rbf_svr.coef_
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
model_svr.coef_
model_svr.kernel
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
from sklearn.model_selectimodel_selection import cross_val_score,ShuffleSplit,LeaveOneout
from sklearn.model_selection import cross_val_score,ShuffleSplit,LeaveOneout
from sklearn.model_selection import cross_val_score,ShuffleSplit,LeaveOneOut
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
cv_split = ShuffleSplit(n_splits=5, train_size=0.75, test_size=0.25)
linear_svr = SVR(kernel = 'linear')
score_ndarray = cross_val_score(linear_svr, X_train, y_train, cv=cv_split)
print(score_ndarray)
param_grid = {
    'svc__cache_size' : [100, 200, 400],
    'svc__C': [1, 10, 100],
    'svc__kernel' : ['rbf', 'linear',"poly"],
    'svc__degree' : [1, 2, 3, 4],
}
linear_svr_y_predict = linear_svr.predict(X_test)
grid = GridSearchCV(linear_svr, param_grid, cv=cv_split)
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np;
from sklearn.model_selection import cross_val_score,ShuffleSplit,LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
param_grid = {
    'svc__cache_size' : [100, 200, 400],
    'svc__C': [1, 10, 100],
    'svc__kernel' : ['rbf', 'linear',"poly"],
    'svc__degree' : [1, 2, 3, 4],
}
pipe_steps = [
    ('svc', SVC())
]
pipeline = Pipeline(pipe_steps)
linear_svr_y_predict = linear_svr.predict(X_test)
grid = GridSearchCV(pipeline, param_grid, cv=cv_split)
grid.fit(X_train, y_train)
runfile('C:/Users/92156/.spyder-py3/支持向量机 交叉验证.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机 交叉验证.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机 交叉验证.py', wdir='C:/Users/92156/.spyder-py3')
parameters = {'kernel':['linear'], 'gamma':np.logspace(-5, 0, num=6, base=2.0),'C':np.logspace(-5, 5, num=11, base=2.0)}

#网格搜索：选择十折交叉验证
svr = SVR()
grid_search = GridSearchCV(svr, parameters, cv=10, n_jobs=4, scoring='mean_squared_error')

grid_search.fit(X_train,y_train)
#linear_svr_y_predict = linear_svr.predict(X_test)
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
print(rbf_svr.best_params_)
print(grid_search.best_params_)
linear_svr = SVR(kernel = 'linear', c=8.0,gamma= 0.03125")
linear_svr = SVR(kernel = 'linear', c=8.0,gamma=("0.03125"))


linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)
linear_svr = SVR(kernel = 'linear', C=8.0,gamma=("0.03125"))
linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)
linear_svr.fit(X_train, y_train)
print(y_text)
print(y_test)
linear_svr.fit(X_train, y_train)
ss=type(X_train)
ss
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
linear_svr = SVR(kernel = 'linear', c=8.0,gamma=("0.03125"))
linear_svr = SVR(kernel = 'linear', C=8.0,gamma=("0.03125"))
linear_svr.fit(X_train, y_train)
parameters = {'kernel':['linear'], 'gamma':np.logspace(-5, 0, num=6, base=2.0),'C':np.logspace(-5, 5, num=11, base=2.0)}

#网格搜索：选择十折交叉验证
svr = SVR()
grid_search = GridSearchCV(svr, parameters, cv=10, n_jobs=4, scoring='mean_squared_error')

grid_search.fit(X_train,y_train)
#linear_svr_y_predict = linear_svr.
print(grid_search.best_score_)
parameters = {'kernel':['linear'], 'gamma':np.logspace(-5, 0, num=6, base=2.0),'C':np.logspace(-5, 5, num=11, base=2.0)}

#网格搜索：选择十折交叉验证
svr = SVR()
grid_search = GridSearchCV(svr, parameters, cv=5, n_jobs=4, scoring='mean_squared_error')

grid_search.fit(X_train,y_train)
linear_svr = SVR(kernel = 'linear', "C"=8.0,"gamma"=0.03125)

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)
from sklearn.svm import SVR

linear_svr = SVR(kernel = 'linear', "C"=8.0,"gamma"=0.03125)

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)
linear_svr = SVR(kernel = 'linear',C=8.0,gamma=0.03125)
linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)
print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/AUC.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/roc.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
model_gbr

mean_tpr = 0.0              # 用来记录画平均ROC曲线的信息
mean_fpr = np.linspace(0, 1, 100)
cnt = 0
for i, (train, test) in enumerate(cv.split(X,y)):       #利用模型划分数据集和目标变量 为一一对应的下标
    cnt +=1
    probas_ = model_gbr.fit(X_train, Y_train).predict_proba(X_text) # 训练模型后预测每条样本得到两种结果的概率
    fpr, tpr, thresholds = roc_curve(y_text, probas_[:, 1])    # 该函数得到伪正例、真正例、阈值，这里只使用前两个
 
    mean_tpr += np.interp(mean_fpr, fpr, tpr)   # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
    mean_tpr[0] = 0.0           # 将第一个真正例=0 以0为起点
 
    roc_auc = auc(fpr, tpr)  # 求auc面积
    plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, roc_auc))    # 画出当前分割数据的ROC曲线
 
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck') # 画对角线
 
mean_tpr /= cnt   # 求数组的平均值
mean_tpr[-1] = 1.0   # 坐标最后一个点为（1,1）  以1为终点
mean_auc = auc(mean_fpr, mean_tpr)
 
plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = {0:.2f})'.format(mean_auc), lw=2)
 
plt.xlim([-0.05, 1.05])     # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')    # 可以使用中文，但需要导入一些库即字体
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

runfile('C:/Users/92156/.spyder-py3/2019D roc.py', wdir='C:/Users/92156/.spyder-py3')
%reset
runfile('C:/Users/92156/.spyder-py3/2019D roc.py', wdir='C:/Users/92156/.spyder-py3')
%reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]  # 去掉了label为2，label只能二分，才可以。
n_samples, n_features = X.shape
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
parameters = {'kernel':['linear'], 'gamma':np.logspace(-5, 0, num=6, base=2.0),'C':np.logspace(-5, 5, num=11, base=2.0)}

#网格搜索：选择十折交叉验证
svr = SVR()
grid_search = GridSearchCV(svr, parameters, cv=5, n_jobs=4, scoring='mean_squared_error')
from sklearn.svm import SVR
parameters = {'kernel':['linear'], 'gamma':np.logspace(-5, 0, num=6, base=2.0),'C':np.logspace(-5, 5, num=11, base=2.0)}

#网格搜索：选择十折交叉验证
svr = SVR()
grid_search = GridSearchCV(svr, parameters, cv=5, n_jobs=4, scoring='mean_squared_error')
from sklearn.model_selection import  GridSearchCV
parameters = {'kernel':['linear'], 'gamma':np.logspace(-5, 0, num=6, base=2.0),'C':np.logspace(-5, 5, num=11, base=2.0)}

#网格搜索：选择十折交叉验证
svr = SVR()
grid_search = GridSearchCV(svr, parameters, cv=5, n_jobs=4, scoring='mean_squared_error')
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV

d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhiPM10"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
grid_search.fit(X_train,y_train)
cv_split = ShuffleSplit(n_splits=10, train_size=0.75, test_size=0.25)
linear_svr = SVR(kernel = 'linear')
score_ndarray = cross_val_score(linear_svr, X_train, y_train, cv=cv_split)
print(score_ndarray)
from sklearn.model_selection import cross_val_score,ShuffleSplit,LeaveOneOut
cv_split = ShuffleSplit(n_splits=10, train_size=0.75, test_size=0.25)
linear_svr = SVR(kernel = 'linear')
score_ndarray = cross_val_score(linear_svr, X_train, y_train, cv=cv_split)
print(score_ndarray)
parameters = {'kernel':['linear'], 'gamma':np.logspace(-5, 0, num=6, base=2.0),'C':np.logspace(-5, 5, num=11, base=2.0)}

#网格搜索：选择十折交叉验证
svr = SVR()
grid_search = GridSearchCV(svr, parameters, cv=5, n_jobs=4, scoring='mean_squared_error')
grid_search.fit(X_train,y_train)
%resset
y
%reset
"""
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV

d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhiPM10"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV

d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhiPM10"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR
parameters = {'kernel':['linear'], 'gamma':np.logspace(-5, 0, num=6, base=2.0),'C':np.logspace(-5, 5, num=11, base=2.0)}
svr = SVR()
grid_search = GridSearchCV(svr, parameters, cv=5, n_jobs=4, scoring='mean_squared_error')
grid_search.fit(X_train,y_train)
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机 交叉验证.py', wdir='C:/Users/92156/.spyder-py3')
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV

d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhico"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

parameters = {"kernel": ("linear", "rbf"), "C": range(1, 100)}
from sklearn import svm
from sklearn import grid_search
from sklearn.datasets import load_iris
iris = load_iris()
svr = svm.SVR()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X_train, y_train)
print (clf.best_params_)    # 最好的参数
%reset
runfile('C:/Users/92156/.spyder-py3/支持向量机 交叉验证.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机 交叉验证.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机 交叉验证.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
%reset
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
import matplotlib.pylab as plt
plt.plot(rbf_svr_y_predict)
plt.plot(rbf_svr_y_predict[100,])
plt.plot(rbf_svr_y_predict[:100,])
plt.plot(rbf_svr_y_predict[:100,],c="r")
plt.plot(rbf_svr_y_predict[:100,],c="r")
plt.plot(y_text[:100,],c="b")

plt.plot(rbf_svr_y_predict[:100,],c="r")
plt.plot(y_test[:100,],c="b")

runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
plt.plot(rbf_svr_y_predict[:100,],c="r")
plt.plot(y_test[:100,],c="b")

runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机 交叉验证.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机 交叉验证.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
plt.plot(rbf_svr_y_predict[:100,],c="r")
plt.plot(y_test[:100,],c="b")
plt.plot(rbf_svr_y_predict[:100,],c="r")
plt.plot(y_test[:100,],c="b")
plt.xlabel("测试集")
plt.ylabel("标准化处理后的数据") 
plt.title("PM2.5数据预测")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(rbf_svr_y_predict[:100,],c="r")
plt.plot(y_test[:100,],c="b")
plt.xlabel("测试集")
plt.ylabel("标准化处理后的数据") 
plt.title("PM2.5数据预测")
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/github  KNN 贝叶斯 SVM 决策树 逻辑回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
plt.plot(pre_svr_y_predict[:100,],c="r")
plt.plot(Y_test[:100,],c="b")
plt.xlabel("测试集")
plt.ylabel("标准化处理后的数据") 
plt.title("PM2.5数据预测")

ss=pre_y_list[5]

plt.plot(ss[:100,],c="r")
plt.plot(Y_test[:100,],c="b")
plt.xlabel("测试集")
plt.ylabel("标准化处理后的数据") 
plt.title("PM2.5数据预测")

ss=pre_y_list[4]

plt.plot(ss[:100,],c="r")
plt.plot(Y_test[:100,],c="b")
plt.xlabel("测试集")
plt.ylabel("标准化处理后的数据") 
plt.title("PM2.5数据预测")

runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
%reset
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
ss=pre_y_list[4]
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
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
y=np.array(d1["chazhio3"])  #选择X值
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]]) #选择Y值
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.15)  #切分数据级
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
        tmp_score = m(y, pre_y_list[i])  # 计算每个回归指标结果
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
y=np.array(d1["chazhio3"])  #选择X值
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]]) #选择Y值
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.15)  #切分数据级
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
df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
for i in range(5):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(y, pre_y_list[i])  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
%reset
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
y=np.array(d1["chazhio3"])  #选择X值
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]]) #选择Y值
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.15)  #切分数据级
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
plt.figure()  # 创建画布
plt.plot(np.arange(X.shape[0]), y, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(X.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线

plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
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
y=np.array(d1["chazhio3"])  #选择X值
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
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
%reset
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
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
y=np.array(d1["chazhiPM2.5"])  #选择X值
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
plt.figure()  # 创建画布
plt.plot(np.arange(X.shape[0]), y, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
plt.figure()  # 创建画布
plt.plot(np.arange(X.shape[0]), y, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(X_train.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线

plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
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
y=np.array(d1["chazhiPM2.5"])  #选择X值
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
plt.plot(np.arange(X.shape[0]), y, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'tan']  # 颜色列表
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
pre_y_list[:,:][:100,]
pre_y_list[3]
for i in range(5):
    print(i)
for i in range(5):
    s.append(pre_y_list[i])
s=[]
for i in range(5):
    s.append(pre_y_list[i])
s=[]
for i in range(5):
    s.append(pre_y_list[i][:100,])
for i in range(5):
    hundred.append(pre_y_list[i][:100,])

plt.figure()  # 创建画布
plt.plot(np.arange(X_test.shape[0]),Y_test, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'tan']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(hundred):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(X_test.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线

plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
hundred=[]  #取qian yibai 
for i in range(5):
    hundred.append(pre_y_list[i][:100,])

plt.figure()  # 创建画布
plt.plot(np.arange(X_test.shape[0]),Y_test, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'tan']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(hundred):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(X_test.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线

plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
hundred=[]  #取qian yibai 
for i in range(5):
    hundred.append(pre_y_list[i][:100,])

plt.figure()  # 创建画布
plt.plot(np.arange(X_test.shape[0]),Y_test, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'tan']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(hundred):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(100), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线

plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
for i, pre_y in enumerate(hundred):  # 读出通过回归模型预测得到的索引及结果
for i, pre_y in enumerate(hundred):  # 读出通过回归模型预测得到的索引及结果
    print(i,pre_y)
    
hundred=[]  #取qian yibai 
for i in range(5):
    hundred.append(pre_y_list[i][:100,])

plt.figure()  # 创建画布
plt.plot(np.arange(X_test.shape[0]),Y_test, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'tan']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(hundred):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(880), pre_y[i], color_list[i], label=model_names[i])  # 画出每条预测结果线

plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
hundred=[]  #取qian yibai 
for i in range(5):
    hundred.append(pre_y_list[i][:100,])

plt.figure()  # 创建画布
plt.plot(np.arange(X_test.shape[0]),Y_test, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'tan']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(hundred):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(100), pre_y[i], color_list[i], label=model_names[i])  # 画出每条预测结果线

plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
plt.figure()  # 创建画布
plt.plot(np.arange(X_test.shape[0]),Y_test[:100,], color='k', label='true y')  # 画出原始值的曲线
hundred=[]  #取qian yibai 
for i in range(5):
    hundred.append(pre_y_list[i][:100,])

plt.figure()  # 创建画布
plt.plot(np.arange(X_test.shape[0]),Y_test[:100,], color='k', label='true y')  # 画出原始值的曲线
hundred=[]  #取qian yibai 
for i in range(5):
    hundred.append(pre_y_list[i][:100,])

plt.figure()  # 创建画布
plt.plot(np.arange(X_test[:100,].shape[0]),Y_test[:100,], color='k', label='true y')  # 画出原始值的曲线
hundred=[]  #取qian yibai 
for i in range(5):
    hundred.append(pre_y_list[i][:100,])

plt.figure()  # 创建画布
plt.plot(np.arange(X_test[:100,].shape[0]),Y_test[:100,], color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'tan']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(hundred):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(X_test[:100,].shape[0]), pre_y[i], color_list[i], label=model_names[i])  # 画出每条预测结果线

plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
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
runfile('C:/Users/92156/.spyder-py3/roc.py', wdir='C:/Users/92156/.spyder-py3')
label=y_text
label=Y_test
pre=pre_y_list[4]
"""
Created on Fri Sep  6 21:15:54 2019

@author: 92156
"""


def AUC(label, pre):
    pos = [i for i in range(len(label)) if label[i] == 1]
    neg = [i for i in range(len(label)) if label[i] == 0]
    
    auc = 0
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5
    
    return auc / (len(pos)*len(neg))



if __name__ == '__main__':

#    label = [1,0,0,0,1,0,1,0]
#    pre = [0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7]
    print(AUC(label, pre))
    
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, th = roc_curve(label, pre , pos_label=1)
    print('sklearn', auc(fpr, tpr))
label.reshape(880,)
label=label.reshape(880,)
def AUC(label, pre):
    pos = [i for i in range(len(label)) if label[i] == 1]
    neg = [i for i in range(len(label)) if label[i] == 0]
    
    auc = 0
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5
    return auc / (len(pos)*len(neg))



if __name__ == '__main__':

#    label = [1,0,0,0,1,0,1,0]
#    pre = [0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7]
    print(AUC(label, pre))
    
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, th = roc_curve(label, pre , pos_label=1)
    print('sklearn', auc(fpr, tpr))
from sklearn.metrics import accuracy_score
accuracy_score(label, pre, normalize=True, sample_weight=None)
from sklearn.metrics import accuracy_score
accuracy_score(label, pre)
from sklearn.metrics import explained_variance_score
explained_variance_score(label, pre)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(label, pre)
from sklearn.metrics import mean_squared_error
mean_squared_error(label, pre)
from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(label, pre)
from sklearn.metrics import median_absolute_error
median_absolute_error(label, pre)
#中位数绝对误差适用于包含异常值的数据的衡量
from sklearn.metrics import r2_score
r2_score(label, pre, multioutput='variance_weighted')
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=5))  # 默认是3-fold cross validation
from sklearn.model_selection import cross_val_score
print(cross_val_score(SVR, X_train, Y_train, cv=5))  # 默认是3-fold cross validation
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X_train, Y_train, cv=5))  # 默认是3-fold cross validation
rbf_svr = SVR(kernel = 'rbf')
print(cross_val_score(rbf_svr, X_train, Y_train, cv=5))  # 默认是3-fold cross validation
cross_val_score(rbf_svr, X_train, Y_train,cv=5,scoring='neg_mean_absolute_error')
import pandas as pd 
d1=pd.read_excel("PM2.5得分.xlsx",encoding="gbk")
label=d1["PM2.5y"]
pre=d1["预测PM2.5"]

from sklearn.metrics import explained_variance_score
explained_variance_score(label, pre) 
#这个指标用来衡量我们模型对数据集波动的解释程度，如果取值为1时，模型就完美，越小效果就越差。下面是python的使用情况：
from sklearn.metrics import mean_absolute_error
mean_absolute_error(label, pre)
#给定数据点的平均绝对误差，一般来说取值越小，模型的拟合效果就越好。下面是在python上的实现：
from sklearn.metrics import mean_squared_error
mean_squared_error(label, pre)
#Median absolute error（中位数绝对误差）
from sklearn.metrics import median_absolute_error
median_absolute_error(label, pre)
#中位数绝对误差适用于包含异常值的数据的衡量
#R² score（决定系数、R方）
from sklearn.metrics import r2_score
r2_score(label, pre, multioutput='variance_weighted')
#    其值越接近1，则变量的解释程度就越高，其值越接近0，其解释程度就越弱。
explained_variance_score(label, pre)
mean_absolute_error(label, pre)
r2_score(label, pre, multioutput='variance_weighted')
mean_absolute_error(label, pre)
mean_squared_error(label, pre)
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
import matplotlib.pyplot as plt
d2=pd.read_csv("D12.csv",enconding="gbk")
d2=pd.read_csv("D12.csv",encoding="gbk")
d2=pd.read_csv("913yiwanshang.xlsx",encoding="gbk")
d2=d2[["CO_y","NO2_y","SO2_y","O3_y"]]
d2=pd.read_csv("913yiwanshang.xlsx",encoding="gbk")
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
d2=d1[["CO_y","NO2_y","SO2_y","O3_y"]]
%reset
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
d2=d1[["CO_y","NO2_y","SO2_y","O3_y"]]
d2_x=StandardScaler()
d2=d2_x.fit(d2)
d2_x=StandardScaler()
d2=d2_x.fit(d2)
d2=d2_x.transform(d2)
d2=d1[["CO_y","NO2_y","SO2_y","O3_y"]]
d2=np.array(d2)
d2_x=StandardScaler()
d2=d2_x.fit(d2)
d2=d2_x.transform(d2)
d2
d2_test, d21_test = train_test_split(d2, random_state = 33, test_size = 0.25)
d2=d1[["CO_y","NO2_y","SO2_y","O3_y"]]
d2=np.array(d2)
d2_test, d21_test = train_test_split(d2, random_state = 33, test_size = 0.25)
%reset
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhino2"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
d2=d1[["CO_y","NO2_y","SO2_y","O3_y"]]
d2=np.array(d2)
d2_x=StandardScaler()
d2_test=d2_x.fit(d2)
d2_test
print(d2_test)
d2_test=d2_x.transform(d2_test)
d2_test=d2_x.transform(d2)
%reset
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhino2"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
d2=d1[["CO_y","NO2_y","SO2_y","O3_y"]]
d2=np.array(d2)
d2_test, d21_test = train_test_split(d2, random_state = 33, test_size = 0.25)
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd 
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhino2"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
d2=d1[["CO_y","NO2_y","SO2_y","O3_y"]]
d2=np.array(d2)
d2_x=StandardScaler()
d2=d2_x.transform(d2)
d2_test, d21_test = train_test_split(d2, random_state = 33, test_size = 0.25)
d2=d2_x.fit(d2)
d2=d2_x.transform(d2)
d2=d2_x.transform(d2)
d2=np.array(d2)
d2=d2_x.transform(d2)
d2=np.array(d2)
d2_x=StandardScaler()
d2=d2_x.transform(d2)
%reset
runfile('C:/Users/92156/.spyder-py3/回归模型 评价性能.py', wdir='C:/Users/92156/.spyder-py3')
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd 
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhino2"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
d2=d1[["CO_y","NO2_y","SO2_y","O3_y"]]
d2=np.array(d2)
d2_test, d21_test = train_test_split(d2, random_state = 33, test_size = 0.25)
d2_x=StandardScaler()
d2_=d2_x.transform(d2)
d2_test=d2_x.transform(d2_test)
d2_x=StandardScaler()
%resset
%reset
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd 
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhino2"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
d2=d1[["CO_y","NO2_y","SO2_y","O3_y"]]
d2=np.array(d2)
d2_test, d21_test = train_test_split(d2, random_state = 33, test_size = 0.25)
d2_x=StandardScaler()
d2_test=d2_x.fit(d2_test)
d21_test=d2_x.transform(d21_test)
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd 
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhino2"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
d2=d1["NO2_y"]
#"CO_y","NO2_y","SO2_y","O3_y"]
d2=np.array(d2)
d2_test, d21_test = train_test_split(d2, random_state = 33, test_size = 0.25)
d2_x=StandardScaler()
d2_test=d2_x.fit(d2_test)
d21_test=d2_x.transform(d21_test)
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd 
d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhino2"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
d2=d1["NO2_y"]
#"CO_y","NO2_y","SO2_y","O3_y"]
d2=np.array(d2)
d2_test, d21_test = train_test_split(d2, random_state = 33, test_size = 0.25)
d2_test=np.array(d2_test).reshape(-1,1)
d21_test=np.array(d21_test).reshape(-1,1)
d2_x=StandardScaler()
d2_test=d2_x.fit(d2_test)
d21_test=d2_x.transform(d21_test)
d2_test=d2_x.fit_transform(d2_test)
d2_test, d21_test = train_test_split(d2, random_state = 33, test_size = 0.25)
d2_test=d2_x.fit_transform(d2_test)
d2_test=np.array(d2_test).reshape(-1,1)
d21_test=np.array(d21_test).reshape(-1,1)
d2_test=d2_x.fit_transform(d2_test)
d21_test=d2_x.transform(d21_test)
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd 
from sklearn.svm import SVR

d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhino2"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)
d3=X["NO_x"]
d3=d1["NO_x"]
d3=d1["NO2_x"]
d3_train, d31_train = train_test_split(d3, random_state = 33, test_size = 0.25)
d3=d1["NO2_x"]
d3_train, d31_train = train_test_split(d3, random_state = 33, test_size = 0.25)
d3_train=np.array(d3_train).reshape(-1,1)
d31_train=np.array(d31_train).reshape(-1,1)
d3_x=StandardScaler()
d3_train=d3_x.fit_transform(d3_train)
d31_train=d3_x.transform(d31_train)
qq=rbf_svr_y_predict+d31_train
a = numpy.array(rbf_svr_y_predict)

b = numpy.array(d31_train)

import numpy 
a = numpy.array(rbf_svr_y_predict)

b = numpy.array(d31_train)

qq=a+b

a = numpy.array([1,1,1,1,1,1,1,1,1,1])

b = numpy.array([2,2,2,2,2,2,2,2,2,2])

c = a + b

print(type(c))

print(list(c))

a = numpy.array(rbf_svr_y_predict)

b = numpy.array(d31_train)
c = a + b

print(type(c))

print(list(c))

a.tolist()
a=a.tolist()
b=b.tolist()
qq=a+b

a = numpy.array(rbf_svr_y_predict)

b = numpy.array(d31_train)

c=a-b
a = numpy.array(rbf_svr_y_predict)

b = numpy.array(d31_train).reshape(880,0)

a = numpy.array(rbf_svr_y_predict)

b = numpy.array(d31_train).reshape(880,)

c=a-b
a = numpy.array(rbf_svr_y_predict)

b = numpy.array(d31_train).reshape(880,)
qq=a+b
from sklearn.metrics import explained_variance_score
explained_variance_score(qq, d21_test)
from sklearn.metrics import r2_score
r2_score(qq, d21_test, multioutput='variance_weighted')
from sklearn.metrics import explained_variance_score
explained_variance_score(qq, d21_test) 
#这个指标用来衡量我们模型对数据集波动的解释程度，如果取值为1时，模型就完美，越小效果就越差。
# Mean absolute error（平均绝对误差）
from sklearn.metrics import mean_absolute_error
mean_absolute_error(qq, d21_test)
#给定数据点的平均绝对误差，一般来说取值越小，模型的拟合效果就越好。
#Mean squared error（均方误差）
from sklearn.metrics import mean_squared_error
mean_squared_error(qq, d21_test)
#Median absolute error（中位数绝对误差）
from sklearn.metrics import median_absolute_error
median_absolute_error(qq, d21_test)
#中位数绝对误差适用于包含异常值的数据的衡量
#R² score（决定系数、R方）
from sklearn.metrics import r2_score
r2_score(qq, d21_test, multioutput='variance_weighted')
#其值越接近1，则变量的解释程度就越高，其值越接近0，其解释程度就越弱。
from sklearn.metrics import median_absolute_error
median_absolute_error(qq, d21_test)
#中位数绝对误差适用于包含异常值的数据的衡量
from sklearn.metrics import mean_squared_error
mean_squared_error(qq, d21_test)
#Median absolute error（中位数绝对误差）
from sklearn.metrics import mean_absolute_error
mean_absolute_error(qq, d21_test)
#给定数据点的平均绝对误差，一般来说
from sklearn.metrics import explained_variance_score
explained_variance_score(qq, d21_test) 
#这个指标用来衡量我们模型对数据集波动的解释程度，如果取值为1时，模型就完美，越小效果就越差。
# Mean absolute error（平均绝对误差）
runfile('C:/Users/92156/.spyder-py3/支持向量机 交叉验证.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
import pandas as pd
asd1=pd.read_excel("PM2.5.xlsx")
asd1=pd.read_excel("PM2.5得分.xlsx")
plt.plot()
s=asd1["PM2.5biaozhun"]
ss=asd1["预测PM2.5biaozhun"]
plt.plot(s,ss)  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像

plt.plot(s,c="r")  # 画出每条预测结果线
plt.plot(ss,c="b")
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像

plt.plot(s[:100,],c="r")  # 画出每条预测结果线
plt.plot(ss[:100,],c="b")
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像

plt.plot(s[:100,],c="r")  # 画出每条预测结果线
plt.plot(ss[:100,],c="b")
plt.title('regression result comparison')  # 标题
plt.legend()
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像

plt.plot(s[:100,],c="r",label='PM2.5')  # 画出每条预测结果线
plt.plot(ss[:100,],c="b",label='PM2.5预测')
plt.title('regression result comparison')  # 标题
plt.legend()
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(s[:100,],c="r",label='PM2.5')  # 画出每条预测结果线
plt.plot(ss[:100,],c="b",label='PM2.5预测')
plt.title('regression result comparison')  # 标题
plt.legend()
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像

runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
%reset
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
plt.plot(rbf_svr_y_predict[:100,],c="r",label='NO2预测')
plt.plot(y_test[:100,],c="b",label='NO2')
plt.xlabel("测试集")
plt.ylabel("标准化处理后的数据") 
plt.title("NO2数据预测")
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
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
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
%reset
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
ss=pre_y_list[4]
s=Y_test
plt.plot(s[:100,],c="r",label='PM2.5')  # 画出每条预测结果线
plt.plot(ss[:100,],c="b",label='PM2.5预测')
plt.title('regression result comparison')  # 标题
plt.legend()
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像

s=Y_test
plt.plot(s[:100,],c="r",label='PM10')  # 画出每条预测结果线
plt.plot(ss[:100,],c="b",label='PM10预测')
plt.title('regression result comparison')  # 标题
plt.legend()
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像

runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
ss=pre_y_list[4]
s=Y_test
plt.plot(s[:100,],c="r",label='O3')  # 画出每条预测结果线
plt.plot(ss[:100,],c="b",label='O3预测')
plt.title('regression result comparison')  # 标题
plt.legend()
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像.

runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
ss=pre_y_list[4]
s=Y_test
plt.plot(s[:100,],c="r",label='CO')  # 画出每条预测结果线
plt.plot(ss[:100,],c="b",label='CO预测')
plt.title('regression result comparison')  # 标题
plt.legend()
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像.

runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/svr训练.py', wdir='C:/Users/92156/.spyder-py3')
ss=pre_y_list[4]
s=Y_test
plt.plot(s[:100,],c="r",label='SO2')  # 画出每条预测结果线
plt.plot(ss[:100,],c="b",label='SO2预测')
plt.title('regression result comparison')  # 标题
plt.legend()
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像.

runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
"""

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
#ss_y = StandardScaler()#标准化
Y_train=np.array(Y_train).reshape(-1,1) #转置  不然会报错 一个warning
Y_test=np.array(Y_test).reshape(-1,1)#转置  不然会报错 一个warning 
X_train = ss_X.fit_transform(X_train)  #模型
X_test = ss_X.transform(X_test)  #模型
#Y_train = ss_y.fit_transform(Y_train) #模型
#Y_test = ss_y.transform(Y_test) #模型


# 训练回归模型
n_folds = 8  # 设置交叉检验的次数
#model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
#model_lr = LinearRegression()  # 建立普通线性回归模型对象
#model_etc = ElasticNet()  # 建立弹性网络回归模型对象
#model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['GBR']  # 不同模型的名称列表
model_dic = [model_gbr]  # 不同回归模型对象的集合
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
#ss_y = StandardScaler()#标准化
Y_train=np.array(Y_train).reshape(-1,1) #转置  不然会报错 一个warning
Y_test=np.array(Y_test).reshape(-1,1)#转置  不然会报错 一个warning 
X_train = ss_X.fit_transform(X_train)  #模型
X_test = ss_X.transform(X_test)  #模型
#Y_train = ss_y.fit_transform(Y_train) #模型
#Y_test = ss_y.transform(Y_test) #模型


# 训练回归模型
n_folds = 8  # 设置交叉检验的次数
#model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
#model_lr = LinearRegression()  # 建立普通线性回归模型对象
#model_etc = ElasticNet()  # 建立弹性网络回归模型对象
#model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['GBR']  # 不同模型的名称列表
model_dic = [model_gbr]  # 不同回归模型对象的集合
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
y=np.array(d1["chazhiso2"])  #选择X值
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]]) #选择Y值
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25)  #切分数据级
ss_X = StandardScaler() #标准化
#ss_y = StandardScaler()#标准化
Y_train=np.array(Y_train).reshape(-1,1) #转置  不然会报错 一个warning
Y_test=np.array(Y_test).reshape(-1,1)#转置  不然会报错 一个warning 
X_train = ss_X.fit_transform(X_train)  #模型
X_test = ss_X.transform(X_test)  #模型
#Y_train = ss_y.fit_transform(Y_train) #模型
#Y_test = ss_y.transform(Y_test) #模型


# 训练回归模型
n_folds = 8  # 设置交叉检验的次数
#model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
#model_lr = LinearRegression()  # 建立普通线性回归模型对象
#model_etc = ElasticNet()  # 建立弹性网络回归模型对象
#model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['GBR']  # 不同模型的名称列表
model_dic = [model_gbr]  # 不同回归模型对象的集合
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
@author: 92156
"""

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
y=np.array(d1["chazhiso2"])  #选择X值
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]]) #选择Y值
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25)  #切分数据级
ss_X = StandardScaler() #标准化
#ss_y = StandardScaler()#标准化
Y_train=np.array(Y_train).reshape(-1,1) #转置  不然会报错 一个warning
Y_test=np.array(Y_test).reshape(-1,1)#转置  不然会报错 一个warning 
X_train = ss_X.fit_transform(X_train)  #模型
X_test = ss_X.transform(X_test)  #模型
#Y_train = ss_y.fit_transform(Y_train) #模型
#Y_test = ss_y.transform(Y_test) #模型


# 训练回归模型
n_folds = 8  # 设置交叉检验的次数
#model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
#model_lr = LinearRegression()  # 建立普通线性回归模型对象
#model_etc = ElasticNet()  # 建立弹性网络回归模型对象
#model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['GBR']  # 不同模型的名称列表
model_dic = [model_gbr]  # 不同回归模型对象的集合
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
for i in range(1):  # 循环每个模型索引
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
y=np.array(d1["chazhiso2"])  #选择X值
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]]) #选择Y值
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25)  #切分数据级
ss_X = StandardScaler() #标准化
#ss_y = StandardScaler()#标准化
Y_train=np.array(Y_train).reshape(-1,1) #转置  不然会报错 一个warning
Y_test=np.array(Y_test).reshape(-1,1)#转置  不然会报错 一个warning 
X_train = ss_X.fit_transform(X_train)  #模型
X_test = ss_X.transform(X_test)  #模型
#Y_train = ss_y.fit_transform(Y_train) #模型
#Y_test = ss_y.transform(Y_test) #模型


# 训练回归模型
n_folds = 8  # 设置交叉检验的次数
#model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
#model_lr = LinearRegression()  # 建立普通线性回归模型对象
#model_etc = ElasticNet()  # 建立弹性网络回归模型对象
#model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['GBR']  # 不同模型的名称列表
model_dic = [model_gbr]  # 不同回归模型对象的集合
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
for i in range(1):  # 循环每个模型索引
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
y=np.array(d1["chazhiso2"])  #选择X值
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
#model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
#model_lr = LinearRegression()  # 建立普通线性回归模型对象
#model_etc = ElasticNet()  # 建立弹性网络回归模型对象
#model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['GBR']  # 不同模型的名称列表
model_dic = [model_gbr]  # 不同回归模型对象的集合
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
for i in range(1):  # 循环每个模型索引
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
y=np.array(d1["chazhiso2"])  #选择X值
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]]) #选择Y值
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25)  #切分数据级
#ss_X = StandardScaler() #标准化
#ss_y = StandardScaler()#标准化
#Y_train=np.array(Y_train).reshape(-1,1) #转置  不然会报错 一个warning
#Y_test=np.array(Y_test).reshape(-1,1)#转置  不然会报错 一个warning 
#X_train = ss_X.fit_transform(X_train)  #模型
#X_test = ss_X.transform(X_test)  #模型
#Y_train = ss_y.fit_transform(Y_train) #模型
#Y_test = ss_y.transform(Y_test) #模型


# 训练回归模型
n_folds = 8  # 设置交叉检验的次数
#model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
#model_lr = LinearRegression()  # 建立普通线性回归模型对象
#model_etc = ElasticNet()  # 建立弹性网络回归模型对象
#model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['GBR']  # 不同模型的名称列表
model_dic = [model_gbr]  # 不同回归模型对象的集合
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
for i in range(1):  # 循环每个模型索引
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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.9)
%reset
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhio3"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.9)
ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.1)
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhio3"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.1)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("913yiwanshang.xlsx",encoding="gbk")
y=np.array(d1["chazhio3"])
X=np.array(d1[["fensu","yaqiang","jiangshuiliang","wendu","shidu","CO_x","NO2_x","SO2_x","O3_x"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.5)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

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
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
d1=pd.read_csv("D12.csv",encoding="gbk")
sns.boxplot(y="PM2.5",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="PM10",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="CO",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="NO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="SO2",data=d1,width=0.3,palette="Blues")
sns.boxplot(y="O3",data=d1,width=0.3,palette="Blues")

## ---(Tue Sep 17 19:31:46 2019)---
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
i
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
def main(page):
    url = 'https://list.jd.com/list.html?cat=9987,653,655' + str(page)
    html = request_dandan(url)
    items = parse_result(html) # 解析过滤我们想要的信息
    for item in items:
        write_item_to_file(item)
for i in range(1,26):
      main(i)
      
%reset
def main(page):
    url = 'https://list.jd.com/list.html?cat=9987,653,655' + str(page)
for i in range(1,26):
    main(i)
url
url = 'https://list.jd.com/list.html?cat=9987,653,655' + str(page)
url = 'https://list.jd.com/list.html?cat=9987,653,655' + str(1)
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
if __name__ == "__main__":
    for i in range(1,26):
        main(i)
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
for i in range(1,26):
    main(i)
    url = "https://list.jd.com/list.html?cat=9987,653,655"+ str(page)+"sort=sort_rank_asc&trans=1&JL=6_0_0#J_main" 
    response = requests.get(url)
        if response.status_code == 200:
            print("ss")

## ---(Wed Sep 18 10:37:54 2019)---
runfile('C:/Users/92156/.spyder-py3/局部线性加权.py', wdir='C:/Users/92156/.spyder-py3')
xArr,yArr = pd.read_excel('财政收入.xls')
import pandas as pd
xArr,yArr = pd.read_excel('财政收入.xls')
QQ= pd.read_excel('财政收入.xls')
QQ
runfile('C:/Users/92156/.spyder-py3/局部线性加权.py', wdir='C:/Users/92156/.spyder-py3')
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
url = "http://bang.dangdang.com/books/newhotsales/01.00.00.00.00.00-24hours-0-0-1-"+ str(page)
url = "http://bang.dangdang.com/books/newhotsales/01.00.00.00.00.00-24hours-0-0-1-"+ str(page)
url = "http://bang.dangdang.com/books/newhotsales/01.00.00.00.00.00-24hours-0-0-1-"+ str(2)
def request_dandan(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        return None
html = request_dandan(url)
import requests
import re
import json
html = request_dandan(url)
pattern = re.compile('<li>.*?list_num.*?(\d+).</div>.*?<img src="(.*?)".*?class="name".*?title="(.*?)">.*?class="star">.*?class="tuijian">(.*?)</span>.*?class="publisher_info">.*?target="_blank">(.*?)</a>.*?class="biaosheng">.*?<span>(.*?)</span></div>.*?<p><span\sclass="price_n">&yen;(.*?)</span>.*?</li>',re.S)
items = re.findall(pattern,html)
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
%reset
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
item
items
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
%reset
url = "http://bang.dangdang.com/books/newhotsales/01.00.00.00.00.00-24hours-0-0-1-1"
def request_dandan(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        return None
html = request_dandan(url)
import requests
import re
import json
html = request_dandan(url)
    'range': item[0],
    'iamge': item[1],
    'title': item[2],
    'recommend': item[3],
    'author': item[4],
    'times': item[5],
    'price': item[6]
}
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
%reset
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
url = "http://bang.dangdang.com/books/newhotsales/01.00.00.00.00.00-24hours-0-0-1-"+ str(page)
html = request_dandan(url)
items = parse_result(html) # 解析过滤我们想要的信息
url = "http://bang.dangdang.com/books/newhotsales/01.00.00.00.00.00-24hours-0-0-1-1"
html = request_dandan(url)
items = parse_result(html) # 解析过滤我们想要的信息
items
print(items)
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
write_item_to_file(item)
for item in items:
    write_item_to_file(item)
str(items)
import re

content = 'Xiaoshuaib has 100 bananas'
res = re.match('^Xi.*(\d+)\s.*s$',content)
print(res.group(1))

import re

content = 'Xiaoshuaib has 100 bananas'
res = re.match('^Xi.*(\d+)\s.*s$',content)
print(res.group(0))

import re

content = 'Xiaoshuaib has 100 bananas'
res = re.match('^Xi.*(\d+)\s.*s$',content)
print(res.group(2))

import re

content = 'Xiaoshuaib has 100 bananas'
res = re.match('^Xi.*(\d+)\s.*s$',content)
print(res.group(1))

import re

content = 'Xiaoshuaib has 100 bananas'
res = re.match('^Xi.*?(\d+)\s.*s$',content)
print(res.group(1))

import re

content = 'Xiaoshuaib has 100 bananas'
res = re.match('^Xi.*?(\d+)\s.*s$',content)
print(res.group(0))

import re

content = 'Xiaoshuaib has 100 bananas'
res = re.match('^Xi.*?(\d+)\s.*s$',content)
print(res.group(2))

runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
runfile('C:/Users/92156/.spyder-py3/当当网 pa.py', wdir='C:/Users/92156/.spyder-py3')
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
runfile('C:/Users/92156/.spyder-py3/当当网 pa.py', wdir='C:/Users/92156/.spyder-py3')
url = 'http://bang.dangdang.com/books/fivestars/01.00.00.00.00.00-recent30-0-0-1-' + str(page)
html = request_dandan(url)
items = parse_result(html) # 解析过滤我们想要的信息
url = 'http://bang.dangdang.com/books/fivestars/01.00.00.00.00.00-recent30-0-0-1-1' 
html = request_dandan(url)
items = parse_result(html) # 解析过滤我们想要的信息

write_item_to_file(items)
def write_item_to_file(item):
    print('开始写入数据 ====> ' + str(item))
    with open('book.txt', 'a', encoding='UTF-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.close()
write_item_to_file(items)
runfile('C:/Users/92156/.spyder-py3/当当网 pa.py', wdir='C:/Users/92156/.spyder-py3')
url = 'http://bang.dangdang.com/books/fivestars/01.00.00.00.00.00-recent30-0-0-1-' + str(page)
if __name__ == "__main__":
    for i in range(1,1):
        main(i)
items
print(items)
items.gi_running
items.next()
runfile('C:/Users/92156/.spyder-py3/当当网 pa.py', wdir='C:/Users/92156/.spyder-py3')
html="html_doc =
 """

<html><head><title>学习python的正确姿势</title></head>
<body>
<p class="title"><b>小帅b的故事</b></p>

<p class="story">有一天，小帅b想给大家讲两个笑话
<a href="http://example.com/1" class="sister" id="link1">一个笑话长</a>,
<a href="http://example.com/2" class="sister" id="link2">一个笑话短</a> ,
他问大家，想听长的还是短的？</p>

<p class="story">...</p>

"""
"
html="html_doc =

<html><head><title>学习python的正确姿势</title></head>
<body>
<p class="title"><b>小帅b的故事</b></p>

<p class="story">有一天，小帅b想给大家讲两个笑话
<a href="http://example.com/1" class="sister" id="link1">一个笑话长</a>,
<a href="http://example.com/2" class="sister" id="link2">一个笑话短</a> ,
他问大家，想听长的还是短的？</p>

<p class="story">...</p>

"
html="""html_doc =
<html><head><title>学习python的正确姿势</title></head>
<body>
<p class="title"><b>小帅b的故事</b></p>
<p class="story">有一天，小帅b想给大家讲两个笑话
<a href="http://example.com/1" class="sister" id="link1">一个笑话长</a>,
<a href="http://example.com/2" class="sister" id="link2">一个笑话短</a> ,
他问大家，想听长的还是短的？</p>
<p class="story">...</p>
"""
soup = BeautifulSoup(html_doc,'lxml')
import beautifulsoup
soup = BeautifulSoup(html_doc,'lxml')
import bs4
soup = bs4(html_doc,'lxml')
soup = bs4(html,'lxml')
import lxml
import beautifulsoup4
runfile('C:/Users/92156/.spyder-py3/豆瓣 pa.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/当当网 pa.py', wdir='C:/Users/92156/.spyder-py3')

## ---(Sat Sep 21 22:10:48 2019)---
from selenium import webdriver
from selenium import webdriver

driver = webdriver.Chrome()
driver.get("https://www.baidu.com")

input = driver.find_element_by_css_selector('#kw')
input.send_keys("苍老师照片")

button = driver.find_element_by_css_selector('#su')
button.click()

from selenium import webdriver

driver = webdriver.Chrome()
driver.get("https://www.baidu.com")

input = driver.find_element_by_css_selector('#kw')
input.send_keys("苍老师照片")

button = driver.find_element_by_css_selector('#su')
button.click()

from selenium import webdriver

driver = webdriver.Chrome()
driver.get("https://www.baidu.com")

input = driver.find_element_by_css_selector('#kw')
input.send_keys("苍老师照片")

button = driver.find_element_by_css_selector('#su')
button.click()

from selenium import webdriver

driver = webdriver.Chrome()
driver.get("https://www.baidu.com")

input = driver.find_element_by_css_selector('#kw')
input.send_keys("苍老师照片")

button = driver.find_element_by_css_selector('#su')
button.click()

from selenium import webdriver


driver = webdriver.Chrome()
driver.get("https://www.baidu.com")

input = driver.find_element_by_css_selector('#kw')
input.send_keys("苍老师照片")

button = driver.find_element_by_css_selector('#su')
button.click()

dr = webdriver.Chrome()
from selenium import webdriver


driver = webdriver.Chrome()
driver.get("https://www.baidu.com")

input = driver.find_element_by_css_selector('#kw')
input.send_keys("苍老师照片")

button = driver.find_element_by_css_selector('#su')
button.click()

login_form = driver.find_element_by_id('loginForm')
driver.current_url
driver.get_cookies()
input.text
driver.page_source

## ---(Sat Sep 21 23:15:15 2019)---
runfile('C:/Users/92156/.spyder-py3/当当网 pa.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/heyun.py', wdir='C:/Users/92156/.spyder-py3')
item
runfile('C:/Users/92156/.spyder-py3/heyun.py', wdir='C:/Users/92156/.spyder-py3')
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
response = requests.get(url)
if response.status_code == 200:
    print("11")
    return response.text
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
response = requests.get(url)
if response.status_code == 200:
    print("11")
        return response.text
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
response = requests.get(url)
if response.status_code == 200:
    print("11")
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
try:
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
except requests.RequestException:
    return None
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
try:
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
response = requests.get(url)
if response.status_code == 200:
    return response.text
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
response = requests.get(url)
if response.status_code == 200:
    print("11")

return response.text
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
response = requests.get(url)
html=response.txt
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
pattern = re.compile("<h2>.*?href=(.*?)",re.S)
items = re.findall(pattern,html)
"""
Created on Sat Sep 21 23:22:39 2019

@author: 92156
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:36:50 2019

@author: 92156
"""

import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
pattern = re.compile("<h2>.*?href=(.*?).*?</a>",re.S)
items = re.findall(pattern,html)
"""
Created on Wed Sep 18 16:36:50 2019

@author: 92156
"""

import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
pattern = re.compile('<article>.*?<h2>.*?href="(.*?)".*?</a>',re.S)
items = re.findall(pattern,html)
"""
Created on Sat Sep 21 23:22:39 2019

@author: 92156
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:36:50 2019

@author: 92156
"""

import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
pattern = re.compile('<article>.*?<a>.*?href="(.*?)".*?</a>',re.S)
items = re.findall(pattern,html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
pattern = re.compile('<article>.*?class="excerpt excerpt-1".*?<a>.*?href="(.*?)".*?',re.S)
items = re.findall(pattern,html)
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
for i in range(15):
    class=excerpt excerpt- +str(i)
for i in range(15):
    aclass=excerpt excerpt- +str(i)
for i in range(15):
    aclass="excerpt excerpt-" +str(i)
for i in range(1,16):
    aclass="excerpt excerpt-" +str(i)
for i in range(1,16):
    aclass="excerpt excerpt-" +str(i)
    pattern = re.compile('article class='+aclass,re.S)
items = re.findall(pattern,html)
pattern
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text

#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<article class=excerpt excerpt-1>.*?',re.S)
items = re.findall(pattern,html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text

#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<article class="excerpt excerpt-1">.*?',re.S)
items = re.findall(pattern,html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text

#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<article class="excerpt excerpt-1">.*?href="(.*?)"',re.S)
items = re.findall(pattern,html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
for i in range(1,16):
    pattern = re.compile('<article class="excerpt excerpt-"+str(i)>.*?href="(.*?)"',re.S)
    items .append( re.findall(pattern,html))
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div> class="content-wrap".*?href="(.*?)"',re.S)
items .append( re.findall(pattern,html))
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div> class="content-wrap"',re.S)
items .append( re.findall(pattern,html))
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div class="catleader">.*?href="(.*?)"',re.S)
items .append( re.findall(pattern,html))
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div class="content">.*?href="(.*?)"',re.S)
items.append( re.findall(pattern,html))
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div class="content">.*?href="(.*?)"')
items.append( re.findall(pattern,html))
"""

import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div class="content">.*?href="(.*?)"',re.S)
items.append( re.findall(pattern,html))
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div class="content">.*?href="(.*?)"',re.S)
items.append( re.findall(pattern,html))
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div class="content">.*?class="focus".*?href="(.*?)"',re.S)
items.append( re.findall(pattern,html))
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div class="content">.*?class="focus".*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
runfile('C:/Users/92156/.spyder-py3/豆瓣 pa.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/heyun soup.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/豆瓣 pa.py', wdir='C:/Users/92156/.spyder-py3')
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div class="content">.*?<h2>.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/weixin/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div class="content">.*?<h2>.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)

ss = 'adafasw12314egrdf5236qew'
num = re.findall('\d+',ss)

import re

ss = 'adafasw12314egrdf5236qew'
num = re.findall('\d+',ss)


ss = 'adafasw12314egrdf5236qew'
num = re.findall('\d+')


ss = 'adafasw12314egrdf5236qew'
num = re.findall('\d+',ss)

import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
html=response.text
num = re.findall('\d+',html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div class="content-wrap">.*?<h2>.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<div class="content">.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<article.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
import requests
#下载地址
for i in item:
    Download_addres=i
#把下载地址发送给requests模块
    f=requests.get(Download_addres)
#下载文件
with open("12.ipg","wb") as code:
     code.write(f.content)
     
import requests
#下载地址
for i in item:
    Download_addres=i
#把下载地址发送给requests模块
    f=requests.get(Download_addres)
#下载文件
with open("12.zip","wb") as code:
     code.write(f.content)
     
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/shangcheng/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<article.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)

from selenium.webdriver.common.keys import Keys    #模仿键盘,操作下拉框的
from bs4 import BeautifulSoup    #解析html的
from selenium import webdriver    #模仿浏览器的


driver = webdriver.Chrome
driver.get(url)#打开你的访问地址
driver.maximize_window()#将页面最大化

driver.find_element_by_xpath('//input[@class="readerImg"]').send_keys(Keys.HOME)#下拉条置顶

url = "https://bbs.5g-yun.com/yuanma/shangcheng/"

driver = webdriver.Chrome
driver.get(url)#打开你的访问地址
driver.maximize_window()#将页面最大化

driver.find_element_by_xpath('//input[@class="readerImg"]').send_keys(Keys.HOME)#下拉条置顶


driver = webdriver.Chrome
driver.get()#打开你的访问地址
driver.maximize_window()#将页面最大化

driver.find_element_by_xpath('//input[@class="readerImg"]').send_keys(Keys.HOME)#下拉条置顶


driver = webdriver.Chrome
driver.get(,url)#打开你的访问地址
driver.maximize_window()#将页面最大化

driver.find_element_by_xpath('//input[@class="readerImg"]').send_keys(Keys.HOME)#下拉条置顶
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/page/2/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<article.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
%reset
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/page/2/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<article.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<article.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<article.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/paga/2/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<article.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/page/2/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<article.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
import re
import time
import requests


class Getfile(object):  #下载文件

    def __init__(self,url):
        self.url=url

    def getheaders(self):
        try:
            r = requests.head(self.url)
            headers =  r.headers
            return headers
        except:
            print('无法获取下载文件大小')
            exit()

    def getfilename(self):  #获取默认下载文件名
        if 'Content-Disposition' in self.getheaders():
            print self.getheaders()
            file = self.getheaders().get('Content-Disposition')
            filename = re.findall('filename="(.*)"',file)
            if filename:
                print filename
                return filename[0]

    def downfile(self,filename):  #下载文件
        self.r = requests.get(self.url,stream=True)
        with open(filename, "wb") as code:
            for chunk in self.r.iter_content(chunk_size=1024): #边下载边存硬盘
                if chunk:
                    code.write(chunk)
        time.sleep(1)



if __name__ == '__main__':

    url = 'https://nbcache00.baidupcs.com/file/3a5f324073b5e5cf9e55e74165264185?bkt=en-038bee77e'
    filename = Getfile(url).getfilename()
    Getfile(url).downfile(filename)
import re
import time
import requests


class Getfile(object):  #下载文件

    def __init__(self,url):
        self.url=url

    def getheaders(self):
        try:
            r = requests.head(self.url)
            headers =  r.headers
            return headers
        except:
            print('无法获取下载文件大小')
            exit()

    def getfilename(self):  #获取默认下载文件名
        if 'Content-Disposition' in self.getheaders():
            print (self.getheaders())
            file = self.getheaders().get('Content-Disposition')
            filename = re.findall('filename="(.*)"',file)
            if filename:
                print (filename)
                return filename[0]

    def downfile(self,filename):  #下载文件
        self.r = requests.get(self.url,stream=True)
        with open(filename, "wb") as code:
            for chunk in self.r.iter_content(chunk_size=1024): #边下载边存硬盘
                if chunk:
                    code.write(chunk)
        time.sleep(1)



if __name__ == '__main__':

    url = 'https://nbcache00.baidupcs.com/file/3a5f324073b5e5cf9e55e74165264185?bkt=en-038bee77e'
    filename = Getfile(url).getfilename()
    Getfile(url).downfile(filename)

import requests
import re
import json
url = "https://bbs.5g-yun.com/yuanma/page/2/"
response = requests.get(url)
if response.status_code == 200:
    print("11")

html=response.text
items=[]
#for i in range(1,16):
#    aclass="excerpt excerpt-" +str(i)
pattern = re.compile('<article.*?href="(.*?)"',re.S)
item= re.findall(pattern,html)
import requests
import re
import json
for i in range(1,16):
    url = "https://bbs.5g-yun.com/yuanma/page/"+str(i)
    response = requests.get(url)
    if response.status_code == 200:
        print("11")
    html=response.text
    items=[]
    #for i in range(1,16):
    #    aclass="excerpt excerpt-" +str(i)
    pattern = re.compile('<article.*?href="(.*?)"',re.S)
    item= re.findall(pattern,html)
import requests
import re
import json
for i in range(1,2):
    url = "https://bbs.5g-yun.com/yuanma/page/"+str(i)
    response = requests.get(url)
    if response.status_code == 200:
        print("11")
    html=response.text
    items=[]
    #for i in range(1,16):
    #    aclass="excerpt excerpt-" +str(i)
    pattern = re.compile('<article.*?href="(.*?)"',re.S)
    item= re.findall(pattern,html)
import requests
import re
import json
for i in range(1,3):
    url = "https://bbs.5g-yun.com/yuanma/page/"+str(i)
    response = requests.get(url)
    if response.status_code == 200:
        print("11")
    html=response.text
    items=[]
    #for i in range(1,16):
    #    aclass="excerpt excerpt-" +str(i)
    pattern = re.compile('<article.*?href="(.*?)"',re.S)
    item= re.findall(pattern,html)
import requests
import re
import json
item=[]
for i in range(1,3):
    url = "https://bbs.5g-yun.com/yuanma/page/"+str(i)
    response = requests.get(url)
    if response.status_code == 200:
        print("11")
    html=response.text
    #for i in range(1,16):
    #    aclass="excerpt excerpt-" +str(i)
    pattern = re.compile('<article.*?href="(.*?)"',re.S)
    item.append(re.findall(pattern,html))
import requests
import re
import json
item=[]
for i in range(1,4):
    url = "https://bbs.5g-yun.com/yuanma/page/"+str(i)
    response = requests.get(url)
    if response.status_code == 200:
        print("11")
    html=response.text
    #for i in range(1,16):
    #    aclass="excerpt excerpt-" +str(i)
    pattern = re.compile('<article.*?href="(.*?)"',re.S)
    item.append(re.findall(pattern,html))
import requests
import re
import json
item=[]
for i in range(1,10):
    url = "https://bbs.5g-yun.com/yuanma/page/"+str(i)
    response = requests.get(url)
    if response.status_code == 200:
        print("11")
    html=response.text
    #for i in range(1,16):
    #    aclass="excerpt excerpt-" +str(i)
    pattern = re.compile('<article.*?href="(.*?)"',re.S)
    item.append(re.findall(pattern,html))
aa="https://bbs.5g-yun.com/1115.html"
import requests
import re
import time
import json
item=[]
for i in range(1,4):
    url = "https://bbs.5g-yun.com/yuanma/page/"+str(i)
    response = requests.get(url)
    if response.status_code == 200:
        print("11")
    html=response.text
    #for i in range(1,16):
    #    aclass="excerpt excerpt-" +str(i)
    pattern = re.compile('<article.*?href="(.*?)"',re.S)
    item.append(re.findall(pattern,html))

time.sleep(60)
for i in range(5,9):
    url = "https://bbs.5g-yun.com/yuanma/page/"+str(i)
    response = requests.get(url)
    if response.status_code == 200:
        print("11")
    html=response.text
    #for i in range(1,16):
    #    aclass="excerpt excerpt-" +str(i)
    pattern = re.compile('<article.*?href="(.*?)"',re.S)
    item.append(re.findall(pattern,html))
%reset
ss = 'adafasw12314egrdf5236qew'
num = re.findall('\d+',ss)

import requests
import re
import time
import json
item=[]
for i in range(1,4):
    url = "https://bbs.5g-yun.com/yuanma/page/"+str(i)
    response = requests.get(url)
    if response.status_code == 200:
        print("11")
    html=response.text
    #for i in range(1,16):
    #    aclass="excerpt excerpt-" +str(i)
    pattern = re.compile('<article.*?href="(.*?)"',re.S)
    item.append(re.findall(pattern,html))

time.sleep(60)

## ---(Sun Sep 22 09:23:20 2019)---
runfile('C:/Users/92156/.spyder-py3/heyun.py', wdir='C:/Users/92156/.spyder-py3')

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.options import Options as FOptions


options = FOptions()
#此处路径写你下载的geckodirver.exe所在的路径。(linux系统无需加.exe后缀，注意'/'与'\')
brower = webdriver.Chrome


brower.get("https://bbs.5g-yun.com/yuanma/")
#使用xpath选出符合条件的内容
print(brower.find_element_by_xpath('//tr[@id="places_neighbours__row"]/td[@class="w2p_fw"]').text)
#页面源码 相当于requests.get().text
print(brower.page_source)
brower.get("https://bbs.5g-yun.com/yuanma/")
#使用css选择器选出符合条件的内容
print(brower.find_element_by_css_selector("#results").text)
#关闭网页
brower.close()


browser.get("http://www.zhihu.com/explore")

brower.get("http://www.zhihu.com/explore")
runfile('C:/Users/92156/.spyder-py3/获取超链接.py', wdir='C:/Users/92156/.spyder-py3')

import requests
import json
import time

for a in range(5):
    url = 'https://movie.douban.com/j/new_search_subjects?sort=T&range=0,10&tags=青春&start={}'.format(a * 20)
    file = requests.get(url).json()  # 返回的是 json文件所以用 .json()
    time.sleep(2)
#每次加载20个电影信息
    for i in range(20):
        dict = file['data'][i]  # 取出字典中 'data' 下第 [i] 部电影的信息
        urlname = dict['url']
        title = dict['title']
        rate = dict['rate']
        cast = dict['casts']
        print('影名:{}  评分:{}  演员:{}  链接:{}\n'.format(title, rate, '、'.join(cast), urlname))
        

import requests

import json

url = 'https://bbs.5g-yun.com/wp-admin/admin-ajax.php'

wbdata = requests.get(url).text

data = json.loads(wbdata)

data
data["data"]
wbdata = requests.get(url)
data = json.loads(wbdata)
data = json.loads(wbdata).txt
data = json.loads(wbdata).text
data = json.loads(wbdata.text)
import requests

import json

url = 'http://www.toutiao.com/api/pc/focus/'

wbdata = requests.get(url).text

data = json.loads(wbdata)

news = data['data']['pc_feed_focus']

for n in news:

title = n['title']

img_url = n['image_url']

url = n['media_url']

print(url,title,img_url)
import requests

import json

url = 'http://www.toutiao.com/api/pc/focus/'

wbdata = requests.get(url).text

data = json.loads(wbdata)

news = data['data']['pc_feed_focus']

for n in news:
title = ['title']

img_url = ['image_url']

url = ['media_url']

print(url,title,img_url)
import requests

import json

url = 'http://www.toutiao.com/api/pc/focus/'

wbdata = requests.get(url).text

data = json.loads(wbdata)

news = data['data']['pc_feed_focus']
for n in news:

    title = n['title']

    img_url = n['image_url']

    url = n['media_url']

    print(url,title,img_url)
    
# coding：utf-8

import requests

import json

url = 'http://www.toutiao.com/api/pc/focus/'

wbdata = requests.get(url).text

data = json.loads(wbdata)

news = data['data']['pc_feed_focus']

for n in news:

    title = n['title']

    img_url = n['image_url']

    url = n['media_url']

    print(url,title,img_url)
    
runfile('C:/Users/92156/.spyder-py3/获取超链接.py', wdir='C:/Users/92156/.spyder-py3')
from selenium import webdriver

#其他浏览器把Chrome换名就行
#option = webdriver.ChromeOptions()
#option.set_headless() 设置无头浏览器，就是隐藏界面后台运行

driver = webdriver.Chrome() #创建driver实例
#driver = webdriver.Chrome(chrome_options=option)  创建实例并载入option

url = "www.baidu.com"
driver.get(url)
#driver.maximize_window() 最大化窗口
#driver.set_window_size(width,height) 设置窗口大小

print(driver.page_source) #打印网页源码
driver.quit() # 关闭浏览器


## ---(Sun Sep 22 12:33:37 2019)---
#coding:utf-8
from selenium import webdriver
import time
brower = webdriver.Firefox()
brower.get("http://www.baidu.com")

brower.find_element_by_id('kw').send_keys('selenium')
brower.find_element_by_id('su').click()

time.sleep(3)
brower.close()

#coding:utf-8
from selenium import webdriver
import time
brower = webdriver.Firefox()
brower.get("http://www.baidu.com")

brower.find_element_by_id('kw').send_keys('selenium')
brower.find_element_by_id('su').click()

time.sleep(3)
brower.close()

from selenium import webdriver
import time
brower = webdriver.Firefox()
brower.get("http://www.baidu.com")

brower.find_element_by_id('kw').send_keys('selenium')
brower.find_element_by_id('su').click()

time.sleep(3)
brower.close()

from selenium import webdriver
import time
brower = webdriver.Firefox()
brower.get("http://www.baidu.com")

brower.find_element_by_id('kw').send_keys('selenium')
brower.find_element_by_id('su').click()

time.sleep(3)
brower.close()


## ---(Sun Sep 22 13:41:09 2019)---
from selenium import webdriver
import time
brower = webdriver.Firefox()
brower.get("http://www.baidu.com")

brower.find_element_by_id('kw').send_keys('selenium')
brower.find_element_by_id('su').click()

time.sleep(3)
brower.close()

from selenium import webdriver
import time
brower = webdriver.Firefox()

from selenium import webdriver
import time
brower = webdriver.Firefox()

from selenium import webdriver
import time
brower = webdriver.Firefox()

from selenium import webdriver
import time
brower = webdriver.Firefox()

from selenium import webdriver
import time
brower = webdriver.Firefox()

from selenium import webdriver
import time
brower = webdriver.Firefox()
brower.get("http://www.baidu.com")

brower.find_element_by_id('kw').send_keys('selenium')
brower.find_element_by_id('su').click()

time.sleep(3)
brower.close()

url="https://bbs.5g-yun.com/wp-admin/admin-ajax.php"
import requests
jsonData=requests.get(url)
print(jsonData.text)

post_data = {"action: wb_front"
"do: single_dl"
"post_id: 14508"}
post_data
post_data = {"action: wb_front",
"do: single_dl",
"post_id: 14508"}
post_data
import requests
return_data=requests.post("https://bbs.5g-yun.com/14508.html,data=post_data)
print return_data.text
import requests
return_data=requests.post("https://bbs.5g-yun.com/14508.html",data=post_data)
print return_data.text
import requests
return_data=requests.post("https://bbs.5g-yun.com/14508.html",data=post_data)
print (return_data.text)

import requests
return_data=requests.post("https://bbs.5g-yun.com/14508.html",data=post_data)
aa=return_data.text

import requests
return_data=requests.post("https://bbs.5g-yun.com",data=post_data)
aa=return_data.text

import selenium
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
import time

browser = webdriver.Firefox() # Get local session of firefox
browser.get("http://news.sina.com.cn/c/2013-07-11/175827642839.shtml ") # Load page
time.sleep(5) # Let the page load
try:
    element = browser.find_element_by_xpath("//span[contains(@class,'f_red')]") # get element on page
    print element.text # get element text
except NoSuchElementException:
    assert 0, "can't find f_red"
browser.close()
import selenium
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
import time

browser = webdriver.Firefox() # Get local session of firefox
browser.get("http://news.sina.com.cn/c/2013-07-11/175827642839.shtml ") # Load page
time.sleep(5) # Let the page load
try:
    element = browser.find_element_by_xpath("//span[contains(@class,'f_red')]") # get element on page
    print (element.text) # get element text
except NoSuchElementException:
    assert 0, "can't find f_red"
browser.close()


## ---(Mon Sep 23 14:34:35 2019)---
html_doc = """

<html><head><title>学习python的正确姿势</title></head>
<body>
<p class="title"><b>小帅b的故事</b></p>

<p class="story">有一天，小帅b想给大家讲两个笑话
<a href="http://example.com/1" class="sister" id="link1">一个笑话长</a>,
<a href="http://example.com/2" class="sister" id="link2">一个笑话短</a> ,
他问大家，想听长的还是短的？</p>

<p class="story">...</p>

"""
import BeautifulSoup
import beautifulsoup4
runfile('C:/Users/92156/.spyder-py3/beautifulsoup4.py', wdir='C:/Users/92156/.spyder-py3')
soup = BeautifulSoup(html_doc,'lxml')
runfile('C:/Users/92156/.spyder-py3/beautifulsoup4.py', wdir='C:/Users/92156/.spyder-py3')
print(soup.p.string)
print(soup.title.parent.name)
print(soup.a.string)
print(soup.a)
print(soup.find_all("a"))
print(soup.find_all("p"))
print(soup.get_text())
url="https://www.zhihu.com/search?type=content&q=beautifulsoup"
soup = BeautifulSoup(url,'lxml')
import time

def moyu_time(name, delay, counter):
 while counter:
   time.sleep(delay)
   print("%s 开始摸鱼 %s" % (name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
   counter -= 1


if __name__ == '__main__':
 moyu_time('小帅b',1,20)
 
runfile('C:/Users/92156/.spyder-py3/多线程.py', wdir='C:/Users/92156/.spyder-py3')
runfile('D:/谷歌下载/dangdangTop500.py', wdir='D:/谷歌下载')
runfile('C:/Users/92156/.spyder-py3/豆瓣 pa.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/豆瓣 pa.py', wdir='C:/Users/92156/.spyder-py3')
list = soup.find(class_='grid_view').find_all('li')
url = 'https://movie.douban.com/top250?start='+ str(page*25)+'&filter='
runfile('C:/Users/92156/.spyder-py3/豆瓣 pa.py', wdir='C:/Users/92156/.spyder-py3')
main(1)
url = 'https://movie.douban.com/top250?start=1&filter='
html = request_douban(url)
from multiprocessing import Process

​def f(name): 
  print('hello', name)​

if __name__ == '__main__': 
  p = Process(target=f, args=('xiaoshuaib',))
  p.start()
  p.join()
from multiprocessing import Pool
​
def f(x):
    return x*x
​
if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
from multiprocessing import Pool
def f(x):
    return x*x
​
if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
from multiprocessing import Pool
def f(x):
    return x*x
if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
        
import pytesseract
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
​
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

captcha = Image.open("captcha1.png")
result = pytesseract.image_to_string(captcha)
print(result)

captcha = Image.open("captcha1.png")
result = pytesseract.image_to_string(captcha)
print(result)

captcha = Image.open("captcha1.png")

result = pytesseract.image_to_string(captcha)
print(result)

captcha = Image.open("captcha1.png")

result = pytesseract.image_to_string(captcha)
print(result)

captcha = Image.open("captcha1.png")

result = pytesseract.image_to_string(captcha)
runfile('C:/Users/92156/.spyder-py3/豆瓣 pa.py', wdir='C:/Users/92156/.spyder-py3')

## ---(Wed Sep 25 13:49:27 2019)---
import requests
url="https://bbs.5g-yun.com/14625.html"
r=requests.get(url)
r.text
qq=r.text
sss="https://www.baidu.com/sugrec?prod=pc_his&from=pc_web&json=1&sid=1429_21113_20698_29522_29720_29567_29221_28704&hisdata=&bs=5G%E4%BA%91%E6%BA%90%E7%A0%81&csor=0&cb=jQuery11020825256859706383_1569391117630&_=1569391117631"
s=requests.get(ss)
s=requests.get(sss)
s.text
jsijdi="https://bbs.5g-yun.com/wp-admin/admin-ajax.php"
import json
rrrr=requests(jsijdi)
rrrr=requests.get(jsijdi)
rrrr.text
import json
d=json.load(rrrr.text)
d=json.loads(rrrr.text)
d
form mat="""
action: wb_front
do: single_dl
post_id: 14625
"""
formmat="""
action: wb_front
do: single_dl
post_id: 14625
"""
requests.get(url,params=formmat)
r=requests.get(url,params=formmat)
r.text
aa=r.text
aa=json.loads(aa)
import time
import json
import random
import requests
from useragents import ua_list
class TencentSpider(object):
  def __init__(self):
    self.one_url = 'https://careers.tencent.com/tencentcareer/api/post/Query?timestamp=1563912271089&countryId=&cityId=&bgIds=&productId=&categoryId=&parentCategoryId=&attrId=&keyword=&pageIndex={}&pageSize=10&language=zh-cn&area=cn'
    self.two_url = 'https://careers.tencent.com/tencentcareer/api/post/ByPostId?timestamp=1563912374645&postId={}&language=zh-cn'
    self.f = open('tencent.json', 'a') # 打开文件
    self.item_list = [] # 存放抓取的item字典数据
  # 获取响应内容函数
  def get_page(self, url):
    headers = {'User-Agent': random.choice(ua_list)}
    html = requests.get(url=url, headers=headers).text
    html = json.loads(html) # json格式字符串转为Python数据类型
    return html
  # 主线函数: 获取所有数据
  def parse_page(self, one_url):
    html = self.get_page(one_url)
    item = {}
    for job in html['Data']['Posts']:
      item['name'] = job['RecruitPostName'] # 名称
      post_id = job['PostId'] # postId，拿postid为了拼接二级页面地址
      # 拼接二级地址,获取职责和要求
      two_url = self.two_url.format(post_id)
      item['duty'], item['require'] = self.parse_two_page(two_url)
      print(item)
      self.item_list.append(item) # 添加到大列表中
  # 解析二级页面函数
  def parse_two_page(self, two_url):
    html = self.get_page(two_url)
    duty = html['Data']['Responsibility'] # 工作责任
    duty = duty.replace('\r\n', '').replace('\n', '') # 去掉换行
    require = html['Data']['Requirement'] # 工作要求
    require = require.replace('\r\n', '').replace('\n', '') # 去掉换行
    return duty, require
  # 获取总页数
  def get_numbers(self):
    url = self.one_url.format(1)
    html = self.get_page(url)
    numbers = int(html['Data']['Count']) // 10 + 1 # 每页有10个推荐
    return numbers
  def main(self):
    number = self.get_numbers()
    for page in range(1, 3):
      one_url = self.one_url.format(page)
      self.parse_page(one_url)
    # 保存到本地json文件:json.dump
    json.dump(self.item_list, self.f, ensure_ascii=False)
    self.f.close()
if __name__ == '__main__':
  start = time.time()
  spider = TencentSpider()
  spider.main()
  end = time.time()
  print('执行时间:%.2f' % (end - start))
  
data="""action: wb_front
do: single_dl
post_id: 14625"""
r = requests.post('https://bbs.5g-yun.com/14625.html',data=datas)
r = requests.post('https://bbs.5g-yun.com/14625.html',data=data)
r.content
xx=r.content
xx
import requests
import re
import time
import json
base_url = 'https://bbs.5g-yun.com/14625.html'

headers = {
    'Origin': 'https://bbs.5g-yun.com',
    'Referer': 'https://bbs.5g-yun.com/14625.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
}


params = {
    "action": "wb_front",
    "do": "single_dl",
    "post_id": "14625",
}

r=requests.post(base_url,headers=headers,data=params)
r.text
ss=r.text
import requests
import re
import time
import json
base_url = 'https://bbs.5g-yun.com/14625.html'

headers = {
    'Referer': 'https://bbs.5g-yun.com/14625.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
    "Sec-Fetch-Mode":" no-cors"
}


params = {
    "action": "wb_front",
    "do": "single_dl",
    "post_id": "14625",
}

r=requests.get(base_url,headers=headers)
import requests
import re
import time
import json
base_url = 'https://bbs.5g-yun.com/14625.html'

headers = {
    'Referer': 'https://bbs.5g-yun.com/14625.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
}


params = {
    "action": "wb_front",
    "do": "single_dl",
    "post_id": "14625",
}

r=requests.get(base_url,headers=headers)
r.text
ss=r.text
url="https://bbs.5g-yun.com/wp-content/plugins/erphpdown/static/erphpdown.js?ver=9.68"
import json
ss=json.loads(url)
r=requests.get(url)
r
r.text
gg=r.text
json.loads(gg)
import requests
import re
import time
import json
base_url = 'https://bbs.5g-yun.com/14625.html'

headers = {
    'Referer': 'https://bbs.5g-yun.com/14625.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',

}


params = {
    "action": "wb_front",
    "do": "single_dl",
    "post_id": "14625",
}

r=requests.get(base_url,headers=headers)
import requests
import re
import time
import json
base_url = 'https://bbs.5g-yun.com/14625.html'

headers = {
    'Referer': 'https://bbs.5g-yun.com/14625.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',

}


params = {
    "action": "wb_front",
    "do": "single_dl",
    "post_id": "14625",
}

r=requests.get(base_url,headers=headers,date=gg)
import requests
import re
import time
import json
base_url = 'https://bbs.5g-yun.com/14625.html'

headers = {
    'Referer': 'https://bbs.5g-yun.com/14625.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',

}


params = {
    "action": "wb_front",
    "do": "single_dl",
    "post_id": "14625",
}

r=requests.get(base_url,headers=headers,data=gg)
import requests
import re
import time
import json
base_url = 'https://bbs.5g-yun.com/14625.html'

headers = {
    'Referer': 'https://bbs.5g-yun.com/14625.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',

}


params = {
    "action": "wb_front",
    "do": "single_dl",
    "post_id": "14625",
}

r=requests.get(base_url,headers=headers,data=gg.encode("utf-8"))
r.text
url="https://bbs.5g-yun.com/wp-includes/js/wp-embed.min.js?ver=5.2.3"
r=requests.get(url)
gg=r.text
import requests
import re
import time
import json
base_url = 'https://bbs.5g-yun.com/14625.html'

headers = {
    'Referer': 'https://bbs.5g-yun.com/14625.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',

}


params = {
    "action": "wb_front",
    "do": "single_dl",
    "post_id": "14625",
}

r=requests.get(base_url,headers=headers,data=gg.encode("utf-8"))
r.text

## ---(Mon Oct 21 16:42:02 2019)---
import TEDTalks
import TEDSubs
TEDSubs https://www.ted.com/talks/meg_jay_why_30_is_not_the_new_20
TEDSubs

## ---(Sat Nov 23 21:26:17 2019)---
import keras

## ---(Sat Nov 23 22:47:55 2019)---
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)

import torch
torch.cuda.is_available()

torch.cuda.set_device(1)                            #　指定gpu1
from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

rand_x = torch.rand(5, 3)
print(rand_x)

from lifelines.estimation import KaplanMeierFitter
from lifelines import NelsonAalenFitter, CoxPHFitter, KaplanMeierFitter from lifelines.statistics import logrank_test
from lifelines import NelsonAalenFitter, CoxPHFitter, KaplanMeierFitter 
from lifelines.statistics import logrank_test


## ---(Sun Nov 24 09:57:27 2019)---
from lifelines import KaplanMeierFitter

durations = [11, 74, 71, 76, 28, 92, 89, 48, 90, 39, 63, 36, 54, 64, 34, 73, 94, 37, 56, 76]
event_observed = [True, True, False, True, True, True, True, False, False, True, True,
                  True, True, True, True, True, False, True, False, True]

kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed)
kmf.plot()


## ---(Wed Nov 27 22:56:42 2019)---
runfile('C:/Users/92156/Desktop/python_data_analysis_and_mining_action-master/python_data_analysis_and_mining_action-master/chapter3/code.py', wdir='C:/Users/92156/Desktop/python_data_analysis_and_mining_action-master/python_data_analysis_and_mining_action-master/chapter3')
runfile('C:/Users/92156/Desktop/python_data_analysis_and_mining_action-master/python_data_analysis_and_mining_action-master/chapter4/code.py', wdir='C:/Users/92156/Desktop/python_data_analysis_and_mining_action-master/python_data_analysis_and_mining_action-master/chapter4')

## ---(Thu Nov 28 17:46:59 2019)---
runfile('C:/Users/92156/.spyder-py3/temp.py', wdir='C:/Users/92156/.spyder-py3')
runfile('C:/Users/92156/.spyder-py3/2019D.py', wdir='C:/Users/92156/.spyder-py3')

## ---(Fri Nov 29 10:19:51 2019)---
import pandas as pd
cc1=pd.read_excel("hangzhou.xls")

cc2=pd.read_excel("zhibiao.xls")
cc2=cc2[1:]

from sklearn import  preprocessing
x=cc2.iloc[:,2:14]
y=cc2.iloc[:,1]
process_b=preprocessing.scale(x)

model = LinearRegression()
model.fit(x,y)

a  = model.intercept_#截距

b = model.coef_#回归系数

print("最佳拟合线:截距",a,",回归系数：",b)
print(x)
y

import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from sklearn import  preprocessing
x=cc2.iloc[:,2:14]
y=cc2.iloc[:,1]
process_b=preprocessing.scale(x)

model = LinearRegression()
model.fit(x,y)

a  = model.intercept_#截距

b = model.coef_#回归系数

print("最佳拟合线:截距",a,",回归系数：",b)
print(x)
y

model = sm.OLS(y, x)
#数据拟合，生成模型
results = model.fit()
print(results.summary())

x=np.asarray(x)
y=np.asarray(y)

model = sm.OLS(y, x)
#数据拟合，生成模型
results = model.fit()
print(results.summary())

import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
cc2=pd.read_excel("zhibiao.xls")
from sklearn import  preprocessing
x=cc2.iloc[:,2:14]
y=cc2.iloc[:,1]
%reset
clear
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
cc2=pd.read_excel("zhibiao.xls")
from sklearn import  preprocessing
x=cc2.iloc[:,2:14]
y=cc2.iloc[:,1]
process_b=preprocessing.scale(x)
model = LinearRegression()
model.fit(process_b,y)
a  = model.intercept_#截距

b = model.coef_#回归系数
print("最佳拟合线:截距",a,",回归系数：",b)
print(x)
y
model = sm.OLS(y, process_b)
results = model.fit()
print(results.summary())
model = sm.OLS(y, x)
#数据拟合，生成模型
results = model.fit()
print(results.summary())
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from factor_analyzer import FactorAnalyzer

cc2
x

fa = FactorAnalyzer()
fa.analyze(x, 5, rotation=None)#固定公共因子个数为5个
print("公因子方差:\n", fa.get_communalities())#公因子方差
print("\n成分矩阵:\n", fa.loadings)#成分矩阵
var = fa.get_factor_variance()#给出贡献率
print("\n解释的总方差（即贡献率）:\n", var)

fa = FactorAnalyzer()

import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(x)
chi_square_value, p_value

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(x)

kmo_model
fa = FactorAnalyzer()
fa.analyze(x, 25, rotation=None)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev


from factor_analyzer import FactorAnalyzer

f=FactorAnalyzer()
f.fit(x)
fa = FactorAnalyzer(5, rotation="varimax")
fa.fit(df)

fa = FactorAnalyzer(5, rotation="varimax")
fa.fit(x)

fa.loadings_

import seaborn as sns
df_cm = pd.DataFrame(np.abs(fa.loadings_), index=x.columns)
plt.figure(figsize = (14,14))
ax = sns.heatmap(df_cm, annot=True, cmap="BuPu")
# 设置y轴的字体的大小
ax.yaxis.set_tick_params(labelsize=15)
plt.title('Factor Analysis', fontsize='xx-large')
# Set y-axis label
plt.ylabel('Sepal Width', fontsize='xx-large')
plt.savefig('factorAnalysis.png', dpi=500)

%reset
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

cc1=pd.read_excel("shiyan.xls")
cc1=pd.read_excel("shiyan.xlsx")
x=cc1.iloc[:,1:6]
y=cc1.iloc[:,0]

x=cc1.iloc[:,1:7]
y=cc1.iloc[:,0]

model = LinearRegression()
model.fit(x,y)

a  = model.intercept_#截距

b = model.coef_#回归系数

print("最佳拟合线:截距",a,",回归系数：",b)
model = sm.OLS(y, x)
#数据拟合，生成模型
results = model.fit()
print(results.summary())
import statsmodels.api as sm
model = sm.OLS(y, x)
#数据拟合，生成模型
results = model.fit()
print(results.summary())

## ---(Sun Dec  1 14:23:47 2019)---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
cc1=pd.read_excel("旅游.xlsx")
import statsmodels.api as sm
cc1
model=LinearRegression()
model


x=cc1.iloc[:,1]
y=cc1.iloc[:,0]
print(x)
y
for i in y:
    z=i+1
z

x=cc1.iloc[:,1]
y=cc1.iloc[:,0]
print(x)
y
z=[]
for i in range(10):
    z.append[2008+i]
z

x=cc1.iloc[:,1]
y=cc1.iloc[:,0]
print(x)
y
z=[]
for i in range(10):
    z.append["2008+i"]
z

x=cc1.iloc[:,1]
y=cc1.iloc[:,0]
print(x)
y
z=[]
for i in range(10):
    z.append(2008+i)
z

x=cc1.iloc[:,1]
y=cc1.iloc[:,0]
print(x)
y
z=[]
for i in range(10):
    z.append(2009+i)
z

linreg = LinearRegression()
model=linreg.fit(x, y)   
import numpy as np
x=np.array(x).reshape(-1,1)
y=np.array(y).reshape(-1,1)

model=linreg.fit(x, y)
xx=model.predict(z)
z=np.array(z).reshape(-1,1)
xx=model.predict(z)
model=linreg.fit(y,x)
xx=model.predict(z)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
cc1=pd.read_excel("旅游.xlsx")
import statsmodels.api as sm
x=cc1.iloc[:,1]
y=cc1.iloc[:,0]

x=np.array(x).reshape(-1,1)
y=np.array(y).reshape(-1,1)

z=[]
for i in range(10):
    z.append(i+2011)
z

model=linreg.fit(y,x)
xx=model.predict(z)

z=np.array(z).reshape(-1,1)
model=linreg.fit(y,x)
xx=model.predict(z)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
cc1=pd.read_excel("旅游.xlsx")
import statsmodels.api as sm
%reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
cc1=pd.read_excel("旅游.xlsx")
import statsmodels.api as sm

## ---(Sat Dec 14 18:16:38 2019)---
import tensorflow as tf
import keras

## ---(Sat Feb 22 15:28:19 2020)---
runfile('C:/Users/92156/Desktop/tt_fund-master/tt_fund-master/tt_fund/spiders/fund_earning.py', wdir='C:/Users/92156/Desktop/tt_fund-master/tt_fund-master/tt_fund/spiders')
fund_type = re.findall(r'kf&ft=(.*?)&rs=&gs=0&sc=zzf&st=desc', response.url)[0]



## ---(Mon Mar  9 15:33:16 2020)---
x = 0o1010
print(x)

x=10
y=3
print(divmod(x,y))

for s in "HelloWorld":
    if s=="W":
        continue
    print(s,end="")
    
d ={"大海":"蓝色", "天空":"灰色", "大地":"黑色"}
print(d["大地"], d.get("大地", "黄色"))

x=（（3**4+5*6**7）/8）**0.5
x=(3**4+5*6**7)/8
x
x**0.5
x = pow((3**4 + 5*(6**7))/8, 0.5)
print("{:.3f}".format(x))

import jieba
s = "中国特色社会主义进入新时代，我国社会主要矛盾已经转化为人民日益增长的美好生活需要和不平衡不充分的发展之间的矛盾。"
n = len(s) 
m = len(jieba.lcut(s))
print("中文字符数为{}，中文词语数为{}。".format(n, m))

print("二进制{0:b}、十进制{0}、八进制{0:o}、十六进制{0:x}".format(0x4DC0+50))
s=[1,"kate",True]
s[3]
s[2]
s[0]
try:
   n = 0
   n = input("请输入一个整数: ")
   def pow10(n):
      return n**10
except:
   print("程序执行错误")
   
if
   n = 0
   n = input("请输入一个整数: ")
   def pow10(n):
      return n**10
except:
   print("程序执行错误")
txt = open("book.txt", "r")
print(txt)
txt.close()


## ---(Thu Mar 19 12:39:20 2020)---
runfile('D:/anaconda/Lib/tt_fund/spiders/ren.py', wdir='D:/anaconda/Lib/tt_fund/spiders')

## ---(Fri Mar 20 20:47:00 2020)---
import pandas as pd
import numpy as np
import servicemanager
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
aa=pd.read_excel("收益比较.xlsx")
aa.set_index(["日期"],inplace=True)
import seaborn as sns

ax = sns.lineplot(data=aa)
plt.figure(figsize=(20,6))
plt.show()

ax = sns.lineplot(data=aa)
plt.figure(figsize=(20,6))
plt.savefig('fix.jpg', dpi=500) #指定分辨率保存
plt.show()

ax = sns.lineplot(data=aa)
plt.figure(figsize=(20,6))

plt.show()
plt.savefig('fix.jpg', dpi=500) #指定分辨率保存


t = np.arange(0,1.1,0.1)

import pandas as pd
import numpy as np
import servicemanager
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
aa=pd.read_excel("收益比较.xlsx")
aa.set_index(["日期"],inplace=True)
import seaborn as sns

ax = sns.lineplot(data=aa)
runfile('C:/Users/92156/.spyder-py3/2009D.py', wdir='C:/Users/92156/.spyder-py3')
import pandas as pd
import numpy as np
import servicemanager
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
aa=pd.read_excel("收益比较.xlsx")
aa.set_index(["日期"],inplace=True)
import seaborn as sns
ax = sns.lineplot(data=aa)
plt.savefig("filename.png")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
aa=pd.read_excel("收益比较.xlsx")
aa.set_index(["日期"],inplace=True)
import seaborn as sns
ax = sns.lineplot(data=aa)
plt.savefig("filename.png")
plt.show()


## ---(Sat Mar 21 12:53:06 2020)---
import numpy as np
import pandas as pd 
answer1=pd.read_excel("因子回测结果(1)(1).xlsx")


answer1
runfile('D:/anaconda/Lib/tt_fund/spiders/ren.py', wdir='D:/anaconda/Lib/tt_fund/spiders')
import numpy as np
import pandas as pd 
answer1=pd.read_excel("因子回测结果(1)(1).xlsx")
answer1.set_index(["yz"], inplace=True)

    
    
  
    
def dataDirection_1(datas):         
    return np.max(datas)-datas     #套公式

#中间型指标 -> 极大型指标
def dataDirection_2(datas, x_best):
    temp_datas = datas - x_best
    M = np.max(abs(temp_datas))
    answer_datas = 1 - abs(datas - x_best) / M     #套公式
    return answer_datas
    
#区间型指标 -> 极大型指标
def dataDirection_3(datas, x_min, x_max):
    M = max(x_min - np.min(datas), np.max(datas) - x_max)
    answer_list = []
    for i in datas:
        if(i < x_min):
            answer_list.append(1 - (x_min-i) /M)      #套公式
        elif( x_min <= i <= x_max):
            answer_list.append(1)
        else:
            answer_list.append(1 - (i - x_max)/M)
    return np.array(answer_list)   
def temp2(datas):
    K = np.power(np.sum(pow(datas,2),axis =1),0.5)
    for i in range(0,K.size):
        for j in range(0,datas[i].size):
            datas[i,j] = datas[i,j] / K[i]      #套用矩阵标准化的公式
    return datas 
def temp3(answer2):
    list_max = np.array([np.max(answer2[0,:]),np.max(answer2[1,:]),np.max(answer2[2,:])])  #获取每一列的最大值
    list_min = np.array([np.min(answer2[0,:]),np.min(answer2[1,:]),np.min(answer2[2,:])])  #获取每一列的最小值
    max_list = []       #存放第i个评价对象与最大值的距离
    min_list = []       #存放第i个评价对象与最小值的距离
    answer_list=[]      #存放评价对象的未归一化得分
    w1,w2,w3,w4=0.7,0.1,0.1,0.1
    for k in range(0,np.size(answer2,axis = 1)):        #遍历每一列数据
        max_sum = 0
        min_sum = 0
        for q in range(0,3):     #有5个指标
            if q== 0 :
                max_sum += w1*np.power(answer2[q,k]-list_max[q],2)     #按每一列计算Di+
                min_sum += w1*np.power(answer2[q,k]-list_min[q],2)     #按每一列计算Di-
            elif q == 1:
                max_sum += w2*np.power(answer2[q,k]-list_max[q],2)     #按每一列计算Di+
                min_sum += w2*np.power(answer2[q,k]-list_min[q],2)     #按每一列计算Di-
            elif q == 2 :
                max_sum += w3*np.power(answer2[q,k]-list_max[q],2)     #按每一列计算Di+
                min_sum += w3*np.power(answer2[q,k]-list_min[q],2)     #按每一列计算Di-
            elif q==3:
                max_sum += w4*np.power(answer2[q,k]-list_max[q],2)     #按每一列计算Di+
                min_sum += w4*np.power(answer2[q,k]-list_min[q],2)     #按每一列计算Di-
            elif q == 4:
                max_sum += w5*np.power(answer2[q,k]-list_max[q],2)     #按每一列计算Di+
                min_sum += w5*np.power(answer2[q,k]-list_min[q],2)     #按每一列计算Di-
        max_list.append(pow(max_sum,0.5))
        min_list.append(pow(min_sum,0.5))
        answer_list.append(min_list[k]/ (min_list[k] + max_list[k]))    #套用计算得分的公式 Si = (Di-) / ((Di+) +(Di-))
        max_sum = 0
        min_sum = 0
    answer = np.array(answer_list)      #得分归一化
    return (answer / np.sum(answer))
def main():
    answer2=[]
    for i in range(0, 4):       #按照不同的列，根据不同的指标转换为极大型指标，因为只有四列
        answer = None
        
        if(i == 3):             #本来就是极大型指标，不用转换
          
            answer = dataDirection_1(answer1["hc"])
         
        else:
            answer = answer1.iloc[:,i]
        answer2.append(answer)
        
         #   print(answer)
        
    #print(answer2)
    answer2 = np.array(answer2)         #将list转换为numpy数组
    answer3 = temp2(answer2)            #数组正向化
    print(answer3)
    answer4 = temp3(answer3)            #标准化处理去钢
    data = pd.DataFrame(answer4)        #计算得分
    print(data)
main()

#	===============RSR ================

import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


def rsr(data, weight=None, threshold=None, full_rank=True):
	Result = pd.DataFrame()
	n, m = data.shape

	# 对原始数据编秩
	if full_rank:
		for i, X in enumerate(data.columns):
			Result[f'X{str(i + 1)}：{X}'] = data.iloc[:, i]
			Result[f'R{str(i + 1)}：{X}'] = data.iloc[:, i].rank(method="dense")
	else:
		for i, X in enumerate(data.columns):
			Result[f'X{str(i + 1)}：{X}'] = data.iloc[:, i]
			Result[f'R{str(i + 1)}：{X}'] = 1 + (n - 1) * (data.iloc[:, i].max() - data.iloc[:, i]) / (data.iloc[:, i].max() - data.iloc[:, i].min())

	# 计算秩和比
	weight = 1 / m if weight is None else pd.np.array(weight) / sum(weight)
	Result['RSR'] = (Result.iloc[:, 1::2] * weight).sum(axis=1) / n
	Result['RSR_Rank'] = Result['RSR'].rank(ascending=False)

	# 绘制 RSR 分布表
	RSR = Result['RSR']
	RSR_RANK_DICT = dict(zip(RSR.values, RSR.rank().values))
	Distribution = pd.DataFrame(index=sorted(RSR.unique()))
	Distribution['f'] = RSR.value_counts().sort_index()
	Distribution['Σ f'] = Distribution['f'].cumsum()
	Distribution[r'\bar{R} f'] = [RSR_RANK_DICT[i] for i in Distribution.index]
	Distribution[r'\bar{R}/n*100%'] = Distribution[r'\bar{R} f'] / n
	Distribution.iat[-1, -1] = 1 - 1 / (4 * n)
	Distribution['Probit'] = 5 - norm.isf(Distribution.iloc[:, -1])

	# 计算回归方差并进行回归分析
	r0 = pd.np.polyfit(Distribution['Probit'], Distribution.index, deg=1)
	print(sm.OLS(Distribution.index, sm.add_constant(Distribution['Probit'])).fit().summary())
	if r0[1] > 0:
		print(f"\n回归直线方程为：y = {r0[0]} Probit + {r0[1]}")
	else:
		print(f"\n回归直线方程为：y = {r0[0]} Probit - {abs(r0[1])}")

	# 代入回归方程并分档排序
	Result['Probit'] = Result['RSR'].apply(lambda item: Distribution.at[item, 'Probit'])
	Result['RSR Regression'] = pd.np.polyval(r0, Result['Probit'])
	threshold = pd.np.polyval(r0, [2, 4, 6, 8]) if threshold is None else pd.np.polyval(r0, threshold)
	Result['Level'] = pd.cut(Result['RSR Regression'], threshold, labels=range(len(threshold) - 1, 0, -1))

	return Result, Distribution


def rsrAnalysis(data, file_name=None, **kwargs):
	Result, Distribution = rsr(data, **kwargs)
	file_name = 'RSR 分析结果报告.xlsx' if file_name is None else file_name + '.xlsx'
	Excel_Writer = pd.ExcelWriter(file_name)
	Result.to_excel(Excel_Writer, '综合评价结果')
	Result.sort_values(by='Level', ascending=False).to_excel(Excel_Writer, '分档排序结果')
	Distribution.to_excel(Excel_Writer, 'RSR分布表')
	Excel_Writer.save()

    
	return Result, Distribution

rsr(answer1)







## ---(Sun Jun 28 14:47:39 2020)---
"""
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("白云机场.xlsx",encoding="gbk")
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("白云机场.xlsx",encoding="gbk")
import pandas as pd
y=np.array(d1["trade_date"])
X=np.array(d1["close"])
d1=pd.read_excel("白云机场.xlsx",encoding="gbk")
y=np.array(d1["trade_date"])
X=np.array(d1["close"])
runfile('C:/Users/92156/.spyder-py3/支持向量机回归.py', wdir='C:/Users/92156/.spyder-py3')
d1=pd.read_excel("白云机场.xlsx",encoding="gbk")
ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
#%画图 PM2.5 /15  00：14
plt.plot(rbf_svr_y_predict[:100,],c="r",label="NO2预测")
plt.plot(y_test[:100,],c="b",label='NO2')
plt.legend()
plt.ylabel("real and predicted value") 
plt.title("regression result comparison")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)
ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
#%画图 PM2.5 /15  00：14
plt.plot(rbf_svr_y_predict[:100,],c="r",label="NO2预测")
plt.plot(y_test[:100,],c="b",label='NO2')
plt.legend()
plt.ylabel("real and predicted value") 
plt.title("regression result comparison")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
x_train=np.array(x_train).reshape(-1,1)
x_test=np.array(x_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
#%画图 PM2.5 /15  00：14
plt.plot(rbf_svr_y_predict[:100,],c="r",label="NO2预测")
plt.plot(y_test[:100,],c="b",label='NO2')
plt.legend()
plt.ylabel("real and predicted value") 
plt.title("regression result comparison")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
X_train=np.array(X_train).reshape(-1,1)
X_test=np.array(X_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
#%画图 PM2.5 /15  00：14
plt.plot(rbf_svr_y_predict[:100,],c="r",label="NO2预测")
plt.plot(y_test[:100,],c="b",label='NO2')
plt.legend()
plt.ylabel("real and predicted value") 
plt.title("regression result comparison")

## ---(Wed Jul  1 17:13:01 2020)---
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame

#导入库


#cc6.to_csv("cc6.csv",encoding="gbk")
#防止输出中文乱码

cc1=pd.read_csv("cc1.csv")
cc2=pd.read_csv("cc2.csv")
cc3=pd.read_csv("cc3.csv")
cc4=pd.read_csv("cc4.csv")
cc1=pd.read_csv("cc1.csv")
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame

#导入库


#cc6.to_csv("cc6.csv",encoding="gbk")
#防止输出中文乱码
cc1=pd.read_csv("cc1.csv")
cc2=pd.read_csv("cc2.csv")
cc3=pd.read_csv("cc3.csv")
cc4=pd.read_csv("cc4.csv")
runfile('C:/Users/92156/.spyder-py3/2018C.py', wdir='C:/Users/92156/.spyder-py3')

## ---(Sun Jul 12 10:19:19 2020)---
import pandas as pd
import xlrd
import xlsxwriter as xw
old_workbook = xlrd.open_workbook('C:/Users/92156/201502.xls')
num_sheets = len(old_workbook.sheets())
cc1=pd.read_excel("二月.xlsx")
cc1=pd.DataFrame(cc1)

num_sheets=cc1.isnull().any(axis=1)

print(num_sheets=cc1.isnull().any(axis=1))


## ---(Sat Aug 15 20:18:29 2020)---
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
cc2=pd.read_excel("财政收入.xls")
from sklearn import  preprocessing
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from sklearn import  preprocessing
cc2=pd.read_excel("财政收入.xls")
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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False #


plt.rcParams['font.family'] = ['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

sns.set_style('whitegrid',{'font.sans-serif':['Arial Unicode MS','Arial']})
cc2=pd.read_excel("财政收入.xls")
runfile('C:/Users/92156/.spyder-py3/temp.py', wdir='C:/Users/92156/.spyder-py3')
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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False #


plt.rcParams['font.family'] = ['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

sns.set_style('whitegrid',{'font.sans-serif':['Arial Unicode MS','Arial']})
cc2=pd.read_excel("财政收入.xls")
runfile('C:/Users/92156/.spyder-py3/temp.py', wdir='C:/Users/92156/.spyder-py3')
ar=np.random.randn(20,4)
df=pd.DataFrame(a,columns=['a','b','c','d'])
df['e']=pd.Series(['one','one','one','one','one','one','two','two','two','two','two','two','two','two',
                   'three','three','three','three','three','three'])
sns.scatterplot(df['a'],df['b'],hue=df['e'])
                   
ar=np.random.randn(20,4)
df=pd.DataFrame(ar,columns=['a','b','c','d'])
df['e']=pd.Series(['one','one','one','one','one','one','two','two','two','two','two','two','two','two',
                   'three','three','three','three','three','three'])
sns.scatterplot(df['a'],df['b'],hue=df['e'])
                   
sns.scatterplot(cc2["财政收入Y"],cc2[["各项税收X1","经济活动人口X2","国民生产总值X3"]])
sns.scatterplot(cc2["财政收入Y"],cc2[["各项税收X1"]])
财政收入Y
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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False #


plt.rcParams['font.family'] = ['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

sns.set_style('whitegrid',{'font.sans-serif':['Arial Unicode MS','Arial']})
cc2=pd.read_excel("财政收入.xls")

sns.scatterplot(cc2["财政收入Y"],cc2[["财政收入Y"]])
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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False #


plt.rcParams['font.family'] = ['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

sns.set_style('whitegrid',{'font.sans-serif':['Arial Unicode MS','Arial']})
cc2=pd.read_excel("财政收入.xls")

sns.pairplot(cc2)
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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False #


plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题

sns.set_style('whitegrid',{'font.sans-serif':['Arial Unicode MS','Arial']})
cc2=pd.read_excel("财政收入.xls")

sns.pairplot(cc2)
runfile('C:/Users/92156/.spyder-py3/temp.py', wdir='C:/Users/92156/.spyder-py3')
cc2=cc2.drop("年份")
cc2=cc2.drop("年份",axis=1)
sns.pairplot(cc2)
from matplotlib.font_manager import FontProperties

myfont=FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf',size=14)

sns.set(font=myfont.get_name())


sns.pairplot(cc2)
cc2.corr()
model = linear_model.LinearRegression()
model.fit(cc2[["各项税收X1","经济活动人口X2","国民生产总值X3"]], cc2["财政收入Y"])
display(model.intercept_)  #截距
display(model.coef_)  #线性模型的系数
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(cc2[["各项税收X1","经济活动人口X2","国民生产总值X3"]], cc2["财政收入Y"])
display(model.intercept_)  #截距
display(model.coef_)  #线性模型的系数
model = sm.OLS( cc2["财政收入Y"],cc2[["各项税收X1","经济活动人口X2","国民生产总值X3"]])
results = model.fit()
print(results.summary())

cc1=pd.read_csv("空置率.csv".enconding="gbk")
cc1=pd.read_csv("空置率.csv")
cc1=pd.read_csv("空置率.csv",encoding="gbk")
cc1=pd.read_csv("空置率.csv",encoding="gbk")
sns.pairplot(cc1)
cc1.corr()
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
ss_x = StandardScaler()
x_train=cc1["平均祖金率"]
y_train=cc1["空置率"]
x_train = ss_x.fit_transform(x_train)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
ss_x = StandardScaler()
x_train=cc1["平均租金率"]
y_train=cc1["空置率"]
x_train = ss_x.fit_transform(x_train)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
x_train = ss_x.fit_transform(x_train.reshape(-1, 1))
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
x_train = ss_x.fit_transform(x_train)
x_train = ss_x.fit_transform(x_train.reshape(-1,1))
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
x_train = ss_x.fit_transform(x_train.values.reshape(-1,1))
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
x_train = ss_x.fit_transform(x_train.values.reshape(-1,1))
y_train = ss_x.fit_transform(y_train.reshape(-1, 1))
x_train = ss_x.fit_transform(x_train.reshape(-1,1))
y_train = ss_x.fit_transform(y_train.reshape(-1, 1))
ss_x = StandardScaler()
x_train=cc1["平均租金率"]
y_train=cc1["空置率"]
model = sm.OLS(y_train,x_train)
results = model.fit()
print(results.summary())
sns.pairplot(cc1)
coef1 = np.polyfit(x,y, 1)
poly_fit1 = np.poly1d(coef1)
plt.plot(x, poly_fit1(x), 'g',label="一阶拟合")
print(poly_fit1)

coef2 = np.polyfit(x,y, 2)
poly_fit2 = np.poly1d(coef2)
plt.plot(x, poly_fit2(x), 'b',label="二阶拟合")
print(poly_fit2)

coef3 = np.polyfit(x,y, 3)
poly_fit3 = np.poly1d(coef3)
plt.plot(x, poly_fit3(x), 'y',label="三阶拟合")
print(poly_fit3)

coef4 = np.polyfit(x,y, 4)
poly_fit4 = np.poly1d(coef4)
plt.plot(x, poly_fit4(x), 'k',label="四阶拟合")
print(poly_fit4)

coef5 = np.polyfit(x,y, 5)
poly_fit5 = np.poly1d(coef5)
plt.plot(x, poly_fit5(x), 'r:',label="五阶拟合")
print(poly_fit5)

plt.scatter(x, y, color='black')
plt.legend(loc=2)
plt.show()
x=cc1["平均租金率"]
y=cc1["空置率"]
coef1 = np.polyfit(x,y, 1)
poly_fit1 = np.poly1d(coef1)
plt.plot(x, poly_fit1(x), 'g',label="一阶拟合")
print(poly_fit1)

coef2 = np.polyfit(x,y, 2)
poly_fit2 = np.poly1d(coef2)
plt.plot(x, poly_fit2(x), 'b',label="二阶拟合")
print(poly_fit2)

coef3 = np.polyfit(x,y, 3)
poly_fit3 = np.poly1d(coef3)
plt.plot(x, poly_fit3(x), 'y',label="三阶拟合")
print(poly_fit3)

coef4 = np.polyfit(x,y, 4)
poly_fit4 = np.poly1d(coef4)
plt.plot(x, poly_fit4(x), 'k',label="四阶拟合")
print(poly_fit4)

coef5 = np.polyfit(x,y, 5)
poly_fit5 = np.poly1d(coef5)
plt.plot(x, poly_fit5(x), 'r:',label="五阶拟合")
print(poly_fit5)

plt.scatter(x, y, color='black')
plt.legend(loc=2)
plt.show()
coef1 = np.polyfit(x,y, 1)
poly_fit1 = np.poly1d(coef1)
plt.plot(x, poly_fit1(x), 'g',label="一阶拟合")
print(poly_fit1)

coef2 = np.polyfit(x,y, 2)
poly_fit2 = np.poly1d(coef2)
plt.plot(x, poly_fit2(x), 'b',label="二阶拟合")
print(poly_fit2)

coef3 = np.polyfit(x,y, 3)
poly_fit3 = np.poly1d(coef3)
plt.plot(x, poly_fit3(x), 'y',label="三阶拟合")
print(poly_fit3)

coef4 = np.polyfit(x,y, 4)
poly_fit4 = np.poly1d(coef4)
plt.plot(x, poly_fit4(x), 'k',label="四阶拟合")
print(poly_fit4)

coef5 = np.polyfit(x,y, 5)
poly_fit5 = np.poly1d(coef5)
plt.plot(x, poly_fit5(x), 'r:',label="五阶拟合")
print(poly_fit5)

plt.scatter(x, y, color='black')
plt.legend(loc=1)
plt.show()
coef1 = np.polyfit(x,y, 1)
poly_fit1 = np.poly1d(coef1)
plt.plot(x, poly_fit1(x), 'g',label="一阶拟合")
print(poly_fit1)

#

plt.scatter(x, y, color='black')
plt.legend(loc=1)
plt.show()
coef2 = np.polyfit(x,y, 2)
poly_fit2 = np.poly1d(coef2)
plt.plot(x, poly_fit2(x), 'b',label="二阶拟合")
print(poly_fit2)



plt.scatter(x, y, color='black')
plt.legend(loc=1)
plt.show()
parameter = np.polyfit(x, y, 3)

print(p.coeffs
parameter = np.polyfit(x, y, 3)

print(p.coeffs)
parameter = np.polyfit(x, y, 3)

print(parameter.coeffs)                #输出拟合的系数，顺序从高阶低阶
parameter = np.polyfit(x, y, 3)

print(parameter.coeffs)
parameter = np.polyfit(x, y, 3)
parameter = np.poly1d(parameter)
print(parameter.coeffs)
plt.plot(x,y,'.',x,parameter,'-r') #'-r'表示用红线画出
x=cc1["平均租金率"]
y=cc1["空置率"]


a=np.polyfit(x,y,2)   #用2次多项式拟合x,y数组
b=np.poly1d(a)        #拟合完成后生成多项式对象
c=b(x)  

plt.scatter(x,y,marker='o',label='original datas')                  #对原始数据做散点图
plt.plot(x,c,ls='--',c='red',label='fitting with second polynomial')#对拟合之后的数据作图
plt.legend()                                                        #给图加上图例
plt.show()
a=np.polyfit(x,y,3)   #用2次多项式拟合x,y数组
b=np.poly1d(a)        #拟合完成后生成多项式对象
c=b(x)  

plt.scatter(x,y,marker='o',label='original datas')                  #对原始数据做散点图
plt.plot(x,c,ls='--',c='red',label='fitting with second polynomial')#对拟合之后的数据作图
plt.legend()                                                        #给图加上图例
plt.show()
a=np.polyfit(x,y,4)   #用2次多项式拟合x,y数组
b=np.poly1d(a)        #拟合完成后生成多项式对象
c=b(x)  

plt.scatter(x,y,marker='o',label='original datas')                  #对原始数据做散点图
plt.plot(x,c,ls='--',c='red',label='fitting with second polynomial')#对拟合之后的数据作图
plt.legend()                                                        #给图加上图例
plt.show()
a=np.polyfit(x,y,5)   #用2次多项式拟合x,y数组
b=np.poly1d(a)        #拟合完成后生成多项式对象
c=b(x)  

plt.scatter(x,y,marker='o',label='original datas')                  #对原始数据做散点图
plt.plot(x,c,ls='--',c='red',label='fitting with second polynomial')#对拟合之后的数据作图
plt.legend()                                                        #给图加上图例
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# 构建噪声数据xu,yu
xu = np.random.rand(50) * 4 * np.pi - 2 * np.pi
def f(x):
    return np.sin(x) + 0.5 * x
yu = f(xu)

plt.figure(figsize=(8, 4))
# 用噪声数据xu,yu，得到拟合多项式系数，自由度为5
reg = np.polyfit(xu, yu, 5)
# 计算多项式的函数值。返回在x处多项式的值，p为多项式系数，元素按多项式降幂排序
ry = np.polyval(reg, xu)
# 原先函数绘制
plt.plot(xu, yu, 'bo', label='f(x)')#蓝色虚线
# 拟合绘制
plt.plot(xu, ry, 'r.', label='regression')#红色点状
plt.legend(loc=0)
plt.show()


import numpy as np
import matplotlib.pyplot as plt


x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

poly = np.polyfit(x_data, y_data, deg = 2)

plt.plot(x_data, y_data, 'o')
plt.plot(x_data, np.polyval(poly, x_data))
plt.show()


import numpy as np
import matplotlib.pyplot as plt


x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

poly = np.polyfit(x_data, y_data, deg = 1)

plt.plot(x_data, y_data, 'o')
plt.plot(x_data, np.polyval(poly, x_data))
plt.show()

x=cc1["平均租金率"]
y=cc1["空置率"]


a=np.polyfit(x,y,5)   #用2次多项式拟合x,y数组
b=np.poly1d(a)        #拟合完成后生成多项式对象
c=b(x)  

plt.scatter(x,y,marker='o')                  #对原始数据做散点图
plt.plot(x,c,ls='--',c='red')#对拟合之后的数据作图
plt.legend()                                                        #给图加上图例
plt.show()
a=np.polyfit(x,y,5)   #用2次多项式拟合x,y数组
b=np.poly1d(a)        #拟合完成后生成多项式对象
c=b(x)  

plt.scatter(x,y,marker='o')                  #对原始数据做散点图
plt.plot(x,c,ls='-',c='red')#对拟合之后的数据作图
plt.legend()                                                        #给图加上图例
plt.show()
a=np.polyfit(x,y,5)   #用2次多项式拟合x,y数组
b=np.poly1d(a)        #拟合完成后生成多项式对象
c=b(x)  

plt.scatter(x,y,marker='o')                  #对原始数据做散点图
plt.plot(x,c,c='red')#对拟合之后的数据作图
plt.legend()                                                        #给图加上图例
plt.show()
a=np.polyfit(x,y,5)   #用2次多项式拟合x,y数组
b=np.poly1d(a)        #拟合完成后生成多项式对象
c=b(x)  


plt.plot(x,c,c='red')#对拟合之后的数据作图
plt.legend()                                                        #给图加上图例
plt.show()

import numpy as np
import matplotlib.pyplot as plt


x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

poly = np.polyfit(x_data, y_data, deg = 1)

plt.plot(x_data, y_data, 'o')
plt.plot(x_data, np.polyval(poly, x_data))
plt.show()


import numpy as np
import matplotlib.pyplot as plt


poly = np.polyfit(x, y, deg = 1)

plt.plot(x, y, 'o')
plt.plot(x, np.polyval(poly, x))
plt.show()


import numpy as np
import matplotlib.pyplot as plt


poly = np.polyfit(x, y, deg = 2)

plt.plot(x, y, 'o')
plt.plot(x, np.polyval(poly, x))
plt.show()


import numpy as np
import matplotlib.pyplot as plt


poly = np.polyfit(x, y, deg = 5)

plt.plot(x, y, 'o')
plt.plot(x, np.polyval(poly, x))
plt.show()


import numpy as np
import matplotlib.pyplot as plt


poly = np.polyfit(x, y, deg = 5)

plt.plot(x, y, 'o')
plt.plot(x, np.polyval(poly, x))
plt.show()

a=np.polyfit(x,y,2)#用2次多项式拟合x，y数组
b=np.poly1d(a)#拟合完之后用这个函数来生成多项式对象
c=b(x)#生成多项式对象之后，就是获取x在这个多项式处的值
plt.scatter(x,y,marker='o',label='original datas')#对原始数据画散点图
plt.plot(x,c,ls='--',c='red',label='fitting with second-degree polynomial')#对拟合之后的数据，也就是x，c数组画图
plt.legend()
plt.show()

cc1.sort_index()

cc1.sort_values()
cc1.sort_index(by=["空置率"])

cc1.reset_index()
cc1

cc1=cc1.sort_index(by=["空置率"])
cc1
cc1=cc1.reset_index()
cc1
cc1.drop("index")
cc1.drop("index",axis=1)
x=cc1["平均租金率"]
y=cc1["空置率"]
a=np.polyfit(x,y,5)   #用2次多项式拟合x,y数组
b=np.poly1d(a)        #拟合完成后生成多项式对象
c=b(x)  


plt.plot(x,c,c='red')#对拟合之后的数据作图
plt.legend()                                                        #给图加上图例
plt.show()
x=cc1["平均租金率"]
y=cc1["空置率"]
a=np.polyfit(x,y,3)#用2次多项式拟合x，y数组
b=np.poly1d(a)#拟合完之后用这个函数来生成多项式对象
c=b(x)#生成多项式对象之后，就是获取x在这个多项式处的值
plt.scatter(x,y,marker='o',label='original datas')#对原始数据画散点图
plt.plot(x,c,ls='--',c='red',label='fitting with second-degree polynomial')#对拟合之后的数据，也就是x，c数组画图
plt.legend()
plt.show()
x=cc1["平均租金率"]
y=cc1["空置率"]
a=np.polyfit(x,y,4)#用2次多项式拟合x，y数组
b=np.poly1d(a)#拟合完之后用这个函数来生成多项式对象
c=b(x)#生成多项式对象之后，就是获取x在这个多项式处的值
plt.scatter(x,y,marker='o',label='original datas')#对原始数据画散点图
plt.plot(x,c,ls='--',c='red',label='fitting with second-degree polynomial')#对拟合之后的数据，也就是x，c数组画图
plt.legend()
plt.show()
x=cc1["平均租金率"]
y=cc1["空置率"]
a=np.polyfit(x,y,5)#用2次多项式拟合x，y数组
b=np.poly1d(a)#拟合完之后用这个函数来生成多项式对象
c=b(x)#生成多项式对象之后，就是获取x在这个多项式处的值
plt.scatter(x,y,marker='o',label='original datas')#对原始数据画散点图
plt.plot(x,c,ls='--',c='red',label='fitting with second-degree polynomial')#对拟合之后的数据，也就是x，c数组画图
plt.legend()
plt.show()

## ---(Mon Aug 17 15:30:41 2020)---
cc3=pd.read_excel("旅游收入数据.xlsx",encoding="gbk")
import pandas as pd 
cc3=pd.read_excel("旅游收入数据.xlsx",encoding="gbk")
cc3.corr()

cc4=cc3.corr()
cc4.to_csv("ss.csv")
import seaborn as sns
sns.heatmap(data=cc4)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(data=cc4)
plt.show()

sns.pairplot(data=cc3)
cc4.drop("year",axis=1)
cc3.drop("year",axis=1)
cc3
from sklearn.model_selection import  train_test_splitfrom sklearn.model_selection import  train_test_splittrain_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,y, random_state=1, train_size=0.6,test_size=0.4)
from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(cc3[["x1","x2","x3","x4","x5","x6"]], cc3["y"], test_size=0.25)
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
import numpy as np
rf.fit(X_train,y_train)#进行模型的训练
rf=RandomForestRegressor()#这里使用了默认的参数设置
rf.fit(X_train,y_train)#进行模型的训练
rf.predict(X_test)
y_test
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())                             


clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())                             

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())


from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
import numpy as np  

from sklearn.datasets import load_iris  
cc3=cc3.drop("year",axis=1)
X_train, X_test, y_train, y_test = train_test_split(cc3[["x1","x2","x3","x4","x5","x6"]], cc3["y"], test_size=0.25)

rf=RandomForestRegressor(n_estimators=100,oob_score=True,random_state=42)#这里使用了默认的参数设置  


#随机挑选两个预测不相同的样本  
rf.fit(X_train,y_train)#进行模型的训练  
rf.predict(X_test)

print(instance)
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
import numpy as np  

from sklearn.datasets import load_iris  
cc3=cc3.drop("year",axis=1)
X_train, X_test, y_train, y_test = train_test_split(cc3[["x1","x2","x3","x4","x5","x6"]], cc3["y"], test_size=0.25)

rf=RandomForestRegressor(n_estimators=100,oob_score=True,random_state=42)#这里使用了默认的参数设置  


#随机挑选两个预测不相同的样本  
rf.fit(X_train,y_train)#进行模型的训练  
rf.predict(X_test)

print(y_test)
import pandas as pd 
cc3=pd.read_excel("旅游收入数据.xlsx",encoding="gbk")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split

sns.heatmap(data=cc4)
plt.show()



from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
import numpy as np  

from sklearn.datasets import load_iris  
cc3=cc3.drop("year",axis=1)
X_train, X_test, y_train, y_test = train_test_split(cc3[["x1","x2","x3","x4","x5","x6"]], cc3["y"], test_size=0.25)

rf=RandomForestRegressor(n_estimators=100,oob_score=True,random_state=42)#这里使用了默认的参数设置  


#随机挑选两个预测不相同的样本  
rf.fit(X_train,y_train)#进行模型的训练  
rf.predict(X_test)

print(y_test)
print(rf.predict(X_test))
roc=roc_auc_score(y,model.oob_prediction_)
print("C-stat: ",roc)

from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y,model.oob_prediction_)
print("C-stat: ",roc)

roc=roc_auc_score(y,rf.oob_prediction_)
print("C-stat: ",roc)

score = numpy.mean(cross_val_score(rf,X,y,cv=5,scoring='accuracy'))
import numpy
score = numpy.mean(cross_val_score(rf,X,y,cv=5,scoring='accuracy'))
rf.feature_importances_
str(rf.feature_importances_)
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
x_train, x_test, y_train, y_test = train_test_split(cc3[["x1","x2","x3","x4","x5","x6"]], cc3["y"], test_size=0.25)
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

## ---(Thu Aug 20 17:01:49 2020)---
runfile('C:/Users/92156/Desktop/untitled1.py', wdir='C:/Users/92156/Desktop')
runfile('C:/Users/92156/.spyder-py3/untitled1.py', wdir='C:/Users/92156/.spyder-py3')
cc1=cc1.set_index(['地区'],inplace=True)
cc1
cc1=pd.read_excel("数据.xlsx",encoding="gbk")
cc1.set_index(['地区'],inplace=True)
data = scale(cc1)
from sklearn.model_selection import train_test_split
from sklearn import cluster

# 创建KMeans模型
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

# 将训练数据' X_train '拟合到模型中，此处没有用到标签数据y_train，K均值聚类一种无监督学习。
clf.fit(data)

#print(clf.cluster_centers_)  # 查看KMeans聚类后的5个质心点的值。
mdl['label'] = clf.labels_  # 对原数据表进行类别标记
c = mdl['label'].value_counts()

print(mdl.values)

print(clf.cluster_centers_)
cc1['label'] = clf.labels_  # 对原数据表进行类别标记
c = cc1['label'].value_counts()

print(cc1.values)

cc2=cc1.value
cc2=cc1.values
cc2=pd.to_Dataframe(cc2)
cc2=pd.Dataframe(cc2)
cc2=pd.DataFrame(cc2)
cc2=pd.to_Dataframe(cc2)
def chooseK(dataSet, i):
    list = []
    for j in range(1, i):
        centroids, clusterAssment = kmeans(dataSet, j)
        sum0 = sum(clusterAssment[:, 1])
        list.append(sum0)
    print(list)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list)
    plt.show()
chooseK(cc1)
chooseK(cc1,2)
# 导入相关模块
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 创建仿真聚类数据集
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

distortions = []
Ks = range(1, 11)

# 为不同的超参数拟合模型
for k in Ks:
    km = KMeans(n_clusters=k,
               init='k-means++',
               n_init=10,
               max_iter=300,
               n_jobs=-1,
               random_state=0)

    km.fit(X)
    distortions.append(km.inertia_) # 保存不同超参数对应模型的聚类偏差

plt.rcParams['font.sans-serif'] = 'SimHei'   
plt.figure('百里希文', figfacecolor='lightyellow')

# 绘制不同超参 K 对应的离差平方和折线图
plt.plot(Ks, distortions,'bo-', mfc='r')
plt.xlabel('簇中心的个数 k')
plt.ylabel('离差平方和')
plt.title('用肘方法确定 kmeans 聚类中簇的最佳数量')

plt.show()

from sklearn.cluster import KMeans
chooseK(cc1,2)
runfile('C:/Users/92156/.spyder-py3/untitled1.py', wdir='C:/Users/92156/.spyder-py3')
cc1
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
clc
clear
clean
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
cc1=pd.read_csv("空置率.csv",encoding="gbk")
sns.pairplot(cc1)
cc1.corr()
import pandas as pd 
cc3=pd.read_excel("旅游收入数据.xlsx",encoding="gbk")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split

sns.heatmap(data=cc4)
plt.show()
import pandas as pd 
cc3=pd.read_excel("旅游收入数据.xlsx",encoding="gbk")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split

sns.heatmap(data=cc3)
plt.show()
import pandas as pd 
cc3=pd.read_excel("旅游收入数据.xlsx",encoding="gbk")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
import pandas as pd
cc3=pd.read_excel("旅游收入数据.xlsx",encoding="gbk")
cc3=pd.read_excel("旅游收入数据.xlsx")
cc2=pd.read_excel("财政收入.xls")
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
runfile('C:/Users/92156/.spyder-py3/temp.py', wdir='C:/Users/92156/.spyder-py3')

## ---(Thu Aug 20 18:29:47 2020)---
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
import pandas as pd 
cc3=pd.read_excel("旅游收入数据.xlsx")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split

sns.heatmap(data=cc3)
plt.show()
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
cc3=cc3.drop("year",axis=1)
x_train, x_test, y_train, y_test = train_test_split(cc3[["x1","x2","x3","x4","x5","x6"]], cc3["y"], test_size=0.25)

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
x_train, x_test, y_train, y_test = train_test_split(cc3[["x1","x2","x3","x4","x5","x6"]], cc3["y"], test_size=0.25)

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
x_train, x_test, y_train, y_test = train_test_split(cc3[["x1","x2","x3","x4","x5","x6"]], cc3["y"], test_size=0.25)

rf=RandomForestRegressor(n_estimators=100,oob_score=True,random_state=42)#这里使用了默认的参数设置  


#随机挑选两个预测不相同的样本  
rf.fit(x_train,y_train)#进行模型的训练  
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
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import Ridge,RidgeCV
model = RidgeCV(alphas=[0.1, 1.0, 10.0])  # 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
model.fit(X_train, y_train)   # 线性回归建模
# print('系数矩阵:\n',model.coef_)
# print('线性回归模型:\n',model)
# print('交叉验证最佳alpha值',model.alpha_)  # 只有在使用RidgeCV算法时才有效
# 使用模型预测
y_predicted = model.predict(X_test)
plt.scatter(X_train, y_train, marker='o',color='green',label='训练数据')

# 绘制散点图 参数：x横轴 y纵轴
plt.scatter(X_test, y_predicted, marker='*',color='blue',label='测试数据')
plt.legend(loc=2,prop=myfont)
plt.plot(X_test, y_predicted,c='r')

# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")

# 显示图形
plt.show()
from sklearn.linear_model import Ridge,RidgeCV
model = RidgeCV(alphas=[0.1, 1.0, 10.0])  # 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
model.fit(X_train, y_train)   # 线性回归建模
# print('系数矩阵:\n',model.coef_)
# print('线性回归模型:\n',model)
# print('交叉验证最佳alpha值',model.alpha_)  # 只有在使用RidgeCV算法时才有效
# 使用模型预测
y_predicted = model.predict(X_test)

# 绘制散点图 参数：x横轴 y纵轴
plt.legend(loc=2,prop=myfont)
plt.plot(X_test, y_predicted,c='r')

# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")

# 显示图形
plt.show()
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
plt.plot(X_test, y_predicted,c='r')

# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")

# 显示图形
plt.show()
X_test
y_predicted
plt.legend(loc=2)
plt.plot(y_test, y_predicted,c='r')

# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")

# 显示图形
plt.show()
print(model.score(X_train, y_train))
print(model.score(y_train, y_predicted))
print(model.score(X_test,y_test))
import sklearn.cluster as sc
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
import sklearn.cluster as sc
model = sc.AgglomerativeClustering(n_clusters=4)
pred_y = model.fit_predict(data)
print(pred_y)
import sklearn.cluster as sc
model = sc.AgglomerativeClustering(n_clusters=3)
pred_y = model.fit_predict(data)
print(pred_y)
from sklearn.cluster import DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
db = DBSCAN(eps=0.3, min_samples=10).fit(data)
ore_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask[db.core_sample_indices_] = Truefrom numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
# 定义数据集
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# 定义模型
model = DBSCAN(eps=0.30, min_samples=9)
# 模型拟合与聚类预测
yhat = model.fit_predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
# 获取此群集的示例的行索引
row_ix = where(yhat == cluster)
# 创建这些样本的散布


pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# 绘制散点图
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
# 定义数据集
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# 定义模型
model = DBSCAN(eps=0.30, min_samples=9)
# 模型拟合与聚类预测
yhat = model.fit_predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
# 获取此群集的示例的行索引
row_ix = where(yhat == cluster)
# 创建这些样本的散布
pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# 绘制散点图
pyplot.show()pyplot.show()
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
# 定义数据集
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# 定义模型
model = DBSCAN(eps=0.30, min_samples=9)
# 模型拟合与聚类预测
yhat = model.fit_predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
# 获取此群集的示例的行索引
row_ix = where(yhat == cluster)
# 创建这些样本的散布
pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# 绘制散点图
pyplot.show()
row_ix=where(yhat == cluster)
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
model = DBSCAN(eps=0.30, min_samples=9)
# 模型拟合与聚类预测
yhat = model.fit_predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:

# 获取此群集的示例的行索引
    row_ix=where(yhat == cluster)
# 创建这些样本的散布
pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# 绘制散点图
pyplot.show()
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
model = DBSCAN(eps=0.30, min_samples=9)
# 模型拟合与聚类预测
yhat = model.fit_predict(data)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:

# 获取此群集的示例的行索引
    row_ix=where(yhat == cluster)
# 创建这些样本的散布
# dbscan 聚类
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
# 定义数据集
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# 定义模型
model = DBSCAN(eps=0.30, min_samples=9)
# 模型拟合与聚类预测
yhat = model.fit_predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
# 获取此群集的示例的行索引
row_ix = where(yhat == cluster)
# 创建这些样本的散布
pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# 绘制散点图
pyplot.show()
# dbscan 聚类
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
# 定义数据集
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# 定义模型
model = DBSCAN(eps=0.30, min_samples=9)
# 模型拟合与聚类预测
yhat = model.fit_predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
# 获取此群集的示例的行索引
    row_ix = where(yhat == cluster)
# 创建这些样本的散布
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# 绘制散点图
    pyplot.show()
    
# dbscan 聚类
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
# 定义数据集
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# 定义模型
model = DBSCAN(eps=0.30, min_samples=9)
# 模型拟合与聚类预测
yhat = model.fit_predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
# 获取此群集的示例的行索引
    row_ix = where(yhat == cluster)
# 创建这些样本的散布
pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# 绘制散点图
pyplot.show()

labels = db.labels_
print('Labels:')
print(labels)
raito=len(labels[labels[:] == -1]) / len(labels)
print('Noise raito:',format(raito, '.2%'))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))

for i in range(n_clusters_):
    print('Cluster ',i,':')
    print(list(X[labels == i].flatten()))
    
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
model = DBSCAN(eps=0.30, min_samples=2)
# 模型拟合与聚类预测
yhat = model.fit_predict(data)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:

# 获取此群集的示例的行索引
    row_ix=where(yhat == cluster)
# 创建这些样本的散布
labels = db.labels_
labels
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
model = DBSCAN(eps=10, min_samples=2)
# 模型拟合与聚类预测
yhat = model.fit_predict(data)

labels = db.labels_ 
cc1['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果
cc1.sort_values('cluster_db')
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
model = DBSCAN(eps=0.003, min_samples=2)
# 模型拟合与聚类预测
yhat = model.fit_predict(data)

labels = db.labels_ 
cc1['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果
cc1.sort_values('cluster_db')
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
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=2)   #降到2维
pca.fit(data)                  #训练
newX=pca.fit_transform(data)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_)  #输出贡献率
print(newX)

## ---(Thu Aug 27 22:15:00 2020)---
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score
from sklearn.model_selection import train_test_split
import numpy as np
#加载数据集
boston=load_boston()
x=boston.data
y=boston.target
cc1=pd.read_csv("机器学习数据集.csv",encoding="gbk")
Created on Thu Aug 27 22:15:25 2020

@author: 92156
"""

from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
#加载数据集
cc1=pd.read_csv("机器学习数据集.csv",encoding="gbk")
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
#加载数据集
cc1=pd.read_csv("机器学习数据集.csv",encoding="gbk")
cc1=pd.read_csv("机器学习数据集.csv")
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
#加载数据集
cc1=pd.read_csv("机器学习数据集.csv",encoding="gbk")
cc1=pd.read_csv("机器学习数据集.csv",encoding="gbk")
cc1=pd.read_csv("ccc.csv",encoding="gbk")
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
#加载数据集
cc1=pd.read_csv("ccc.csv",encoding="gbk")
cc2=cc1[["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10"]]
cc3=cc1["y"]
cc1=pd.read_csv("ccc.csv",encoding="gbk")
X=cc1[["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10"]]
y=cc1["y"]
cc1=pd.read_csv("ccc.csv",encoding="gbk")
x=cc1[["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10"]]
y=cc1["y"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
cc1=pd.read_csv("ccc.csv",encoding="gbk")
x=cc1[["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10"]]
y=cc1["y"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)
y_train = StandardScaler().fit_transform(y_train).ravel()
y_test = StandardScaler().fit_transform(y_test).ravel()

#创建svR实例
svr=SVR(C=1, kernel='rbf', epsilon=0.2)
svr=svr.fit(x_train,y_train)
#预测
svr_predict=svr.predict(x_test)
#评价结果
mae = mean_absolute_error(y_test, svr_predict)
mse = mean_squared_error(y_test, svr_predict)
evs = explained_variance_score(y_test, svr_predict)
r2 = r2_score(y_test, svr_predict)
print("MAE：", mae)
print("MSE：", mse)
print("EVS：", evs)
print("R2：", r2)
C = [0.1, 0.2, 0.5, 0.8, 0.9, 1, 2, 5, 10]
kernel = 'rbf'
gamma = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8]
epsilon = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
# 参数字典
params_dict = {
    'C': C,
    'gamma': gamma,
    'epsilon': epsilon
}

# 创建SVR实例
svr = SVR()

# 网格参数搜索
gsCV = GridSearchCV(
    estimator=svr,
    param_grid=params_dict,
    n_jobs=2,
    scoring='r2',
    cv=6
)
gsCV.fit(x_train, y_train)
# 输出参数信息
print("最佳度量值:", gsCV.best_score_)
print("最佳参数:", gsCV.best_params_)
print("最佳模型:", gsCV.best_estimator_)

# 用最佳参数生成模型
svr = SVR(C=gsCV.best_params_['C'], kernel=kernel, gamma=gsCV.best_params_['gamma'],
          epsilon=gsCV.best_params_['epsilon'])

# 获取在训练集的模型
svr.fit(x_train, y_train)

# 预测结果
svr_predict = svr.predict(x_test)

# 模型评测
mae = mean_absolute_error(y_test, svr_predict)
mse = mean_squared_error(y_test, svr_predict)
evs = explained_variance_score(y_test, svr_predict)
r2 = r2_score(y_test, svr_predict)
print("MAE：", mae)
print("MSE：", mse)
print("EVS：", evs)
print("R2：", r2)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
import numpy as np
C = [0.1, 0.2, 0.5, 0.8, 0.9, 1, 2, 5, 10]
kernel = 'rbf'
gamma = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8]
epsilon = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
# 参数字典
params_dict = {
    'C': C,
    'gamma': gamma,
    'epsilon': epsilon
}

# 创建SVR实例
svr = SVR()

# 网格参数搜索
gsCV = GridSearchCV(
    estimator=svr,
    param_grid=params_dict,
    n_jobs=2,
    scoring='r2',
    cv=6
)
gsCV.fit(x_train, y_train)
# 输出参数信息
print("最佳度量值:", gsCV.best_score_)
print("最佳参数:", gsCV.best_params_)
print("最佳模型:", gsCV.best_estimator_)

# 用最佳参数生成模型
svr = SVR(C=gsCV.best_params_['C'], kernel=kernel, gamma=gsCV.best_params_['gamma'],
          epsilon=gsCV.best_params_['epsilon'])

# 获取在训练集的模型
svr.fit(x_train, y_train)

# 预测结果
svr_predict = svr.predict(x_test)

# 模型评测
mae = mean_absolute_error(y_test, svr_predict)
mse = mean_squared_error(y_test, svr_predict)
evs = explained_variance_score(y_test, svr_predict)
r2 = r2_score(y_test, svr_predict)
print("MAE：", mae)
print("MSE：", mse)
print("EVS：", evs)
print("R2：", r2)

## ---(Thu Sep 10 16:50:42 2020)---
import pandas as pd 

cc1=pd.read_csv("附件_一季度.csv")
import pandas as pd 

cc1=pd.read_csv("附件_一季度.csv",encoding="gbk")
import pandas as pd 

cc1=pd.read_excel("附件_一季度.xlsx")
runfile('C:/Users/92156/Desktop/guosai.py', wdir='C:/Users/92156/Desktop')
cc2=pd.read_csv("附件_二季度.csv")
cc2=pd.read_csv("附件_二季度.csv",encoding="gbk")
cc2=pd.read_csv("附件_二季度.csv",encoding="gbk")
cc3=pd.read_csv("附件_三季度.csv",encoding="gbk")
cc4=pd.read_csv("附件_四季度.csv",encoding="gbk")
import numpy as np
df = pd.concat([cc1, cc2])
df.to_csv("df.csv")
df = pd.concat([cc1, cc2])
df.to_csv("df.csv")
df = pd.concat([cc1, cc2,cc3,cc4])
df.to_csv("df.csv")
cc=df["水表名"]==("校医院南+")
cc=df[["水表名"]=="校医院南+"]
cc=df.loc[df["水表名"]=="校医院南+"]
print(cc)
cc.to_csv("校医院南.csv")
cc=df.loc[df["水表名"]=="校医院南"]
cc.to_csv("XXX校医院.csv")
cc=df.loc[df["水表名"]=="XXX校医院"]
cc.to_csv("XXX校医院.csv")
cc=df.loc[df["水表名"]=="车队+"]
cc.to_csv("车队+.csv")
cc=df.loc[df["水表名"]=="XXX花圃+"]
cc.to_csv("XXX花圃+.csv")
cc=df.loc[df["水表名"]=="XXX成教院XXX分院"]
cc.to_csv("XXX成教院XXX分院.csv")
cc=df.loc[df["水表名"]=="XXX田径场厕所"]
cc.to_csv("XXX田径场厕所.csv")
cc=df.loc[df["水表名"]=="离退休活动室
"]
cc.to_csv("离退休活动室
.csv")
cc=df.loc[df["水表名"]=="离退休活动室"]
cc.to_csv("离退休活动室.csv")
cc=df.loc[df["水表名"]=="XXXL馆"]
cc.to_csv("XXXL馆.csv")
cc=df.loc[df["水表名"]=="XXXL楼"]
cc.to_csv("XXXL楼.csv")
cc=df.loc[df["水表名"]=="XXXS馆"]
cc.to_csv("XXXS馆.csv")
cc=df.loc[df["水表名"]=="XXXK"]
cc.to_csv("XXXK.csv")
cc=df.loc[df["水表名"]=="XXXK酒店"]
cc.to_csv("XXXK酒店.csv")
cc=df.loc[df["水表名"]=="XXX体育馆"]
cc.to_csv("XXX体育馆.csv")
cc=df.loc[df["水表名"]=="XXX干训楼"]
cc.to_csv("XXX干训楼.csv")
cc=df.loc[df["水表名"]=="XXX第八学生宿舍"]
cc.to_csv("XXX第八学生宿舍.csv")
cc=df.loc[df["水表名"]=="XXX8舍热泵"]
cc.to_csv("XXX8舍热泵.csv")
cc=df.loc[df["水表名"]=="XXX第七学生宿舍"]
cc.to_csv("XXX第七学生宿舍.csv")
cc=df.loc[df["水表名"]=="XXXK楼"]
cc.to_csv("XXXK楼.csv")
cc=df.loc[df["水表名"]=="XXX第五食堂"]
cc.to_csv("XXX第五食堂.csv")
cc=df.loc[df["水表名"]=="XXX污水处理"]
cc.to_csv("XXX污水处理.csv")
cc=df.loc[df["水表名"]=="64397副表"]
cc.to_csv("64397副表.csv")
cc=df.loc[df["水表名"]=="XXX第一食堂"]
cc.to_csv("XXX第一食堂.csv")
cc=df.loc[df["水表名"]=="区域4+"]
cc.to_csv("区域4+.csv")
cc=df.loc[df["水表名"]=="XXX第二学生宿舍"]
cc.to_csv("XXX第二学生宿舍.csv")
cc=df.loc[df["水表名"]=="XXX老医务室楼"]
cc.to_csv("XXX老医务室楼.csv")
cc=df.loc[df["水表名"]=="教育超市+"]
cc.to_csv("教育超市+.csv")
cc=df.loc[df["水表名"]=="XXX第一学生宿舍"]
cc.to_csv("XXX第一学生宿舍.csv")
cc=df.loc[df["水表名"]=="东大门大棚+"]
cc.to_csv("东大门大棚+.csv")
cc=df.loc[df["水表名"]=="区域3+"]
cc.to_csv("区域3+.csv")
cc=df.loc[df["水表名"]=="老七楼"]
cc.to_csv("老七楼.csv")
cc=df.loc[df["水表名"]=="XXX第五学生宿舍"]
cc.to_csv("XXX第五学生宿舍.csv")
cc=df.loc[df["水表名"]=="XXX5舍热泵热水"]
cc.to_csv("XXX5舍热泵热水.csv")
cc=df.loc[df["水表名"]=="XXX第四学生宿舍"]
cc.to_csv("XXX第四学生宿舍.csv")
cc=df.loc[df["水表名"]=="茶园+"]
cc.to_csv("茶园+.csv")
cc=df.loc[df["水表名"]=="XXX4舍热泵热水"]
cc.to_csv("XXX4舍热泵热水.csv")
cc=df.loc[df["水表名"]=="XXX第三学生宿舍"]
cc.to_csv("XXX第三学生宿舍.csv")
cc=df.loc[df["水表名"]=="XXX3舍热泵热水"]
cc.to_csv("XXX3舍热泵热水.csv")
cc=df.loc[df["水表名"]=="危险品仓库+"]
cc.to_csv("危险品仓库+.csv")
cc=df.loc[df["水表名"]=="纳米楼厕所+"]
cc.to_csv("纳米楼厕所+.csv")
cc=df.loc[df["水表名"]=="理发店+"]
cc.to_csv("理发店+.csv")
cc=df.loc[df["水表名"]=="XXX第二食堂"]
cc.to_csv("XXX第二食堂.csv")
cc=df.loc[df["水表名"]=="东大门温室"]
cc.to_csv("东大门温室.csv")
cc=df.loc[df["水表名"]=="新留学生楼"]
cc.to_csv("新留学生楼.csv")
cc=df.loc[df["水表名"]=="XXX航空航天"]
cc.to_csv("XXX航空航天.csv")
cc=df.loc[df["水表名"]=="XXX锅炉房"]
cc.to_csv("XXX锅炉房.csv")
cc=df.loc[df["水表名"]=="浴室"]
cc.to_csv("浴室.csv")
cc=df.loc[df["水表名"]=="老医务楼2.3楼+"]
cc.to_csv("老医务楼2.3楼+.csv")
cc=df.loc[df["水表名"]=="XXX老六楼"]
cc.to_csv("XXX老六楼.csv")
cc=df.loc[df["水表名"]=="XXX老五楼"]
cc.to_csv("XXX老五楼.csv")
cc=df.loc[df["水表名"]=="留学生楼（新）"]
cc.to_csv("留学生楼（新）.csv")
cc=df.loc[df["水表名"]=="XXX游泳池"]
cc.to_csv("XXX游泳池.csv")
cc=df.loc[df["水表名"]=="XXX第九学生宿舍"]
cc.to_csv("XXX第九学生宿舍.csv")
cc=df.loc[df["水表名"]=="养殖队6721副表+"]
cc.to_csv("养殖队6721副表+.csv")
cc=df.loc[df["水表名"]=="司法鉴定中心"]
cc.to_csv("司法鉴定中心.csv")
cc=df.loc[df["水表名"]=="XXX国际纳米研究所"]
cc.to_csv("XXX国际纳米研究所.csv")
cc=df.loc[df["水表名"]=="纳米楼4.5楼+"]
cc.to_csv("纳米楼4.5楼+.csv")
cc=df.loc[df["水表名"]=="纳米楼3楼+"]
cc.to_csv("纳米楼3楼+.csv")
cc=df.loc[df["水表名"]=="区域2"]
cc.to_csv("区域2.csv")
cc=df.loc[df["水表名"]=="XXXT馆后平房"]
cc.to_csv("XXXT馆后平房.csv")
cc=df.loc[df["水表名"]=="XXX后勤楼"]
cc.to_csv("XXX后勤楼.csv")
cc=df.loc[df["水表名"]=="校管中心种子楼东+"]
cc.to_csv("校管中心种子楼东+.csv")
cc=df.loc[df["水表名"]=="XXX图书馆"]
cc.to_csv("XXX图书馆.csv")
cc=df.loc[df["水表名"]=="XXX毒物研究所"]
cc.to_csv("XXX毒物研究所.csv")
cc=df.loc[df["水表名"]=="XXX种子楼"]
cc.to_csv("XXX种子楼.csv")
cc=df.loc[df["水表名"]=="区域1（西）"]
cc.to_csv("区域1（西）.csv")
cc=df.loc[df["水表名"]=="XXX大楼厕所西"]
cc.to_csv("XXX大楼厕所西.csv")
cc=df.loc[df["水表名"]=="XXX科学楼"]
cc.to_csv("XXX科学楼.csv")
cc=df.loc[df["水表名"]=="XXX大楼厕所东"]
cc.to_csv("XXX大楼厕所东.csv")
cc=df.loc[df["水表名"]=="XXX中心水池"]
cc.to_csv("XXX中心水池.csv")
cc=df.loc[df["水表名"]=="XXX西大楼"]
cc.to_csv("XXX西大楼.csv")
cc=df.loc[df["水表名"]=="书店+"]
cc.to_csv("书店+.csv")
cc=df.loc[df["水表名"]=="新大门传达室+"]
cc.to_csv("新大门传达室+.csv")
cc=df.loc[df["水表名"]=="养殖馆附房保卫处宿舍+"]
cc.to_csv("养殖馆附房保卫处宿舍+.csv")
cc=df.loc[df["水表名"]=="养殖馆公共厕所+"]
cc.to_csv("养殖馆公共厕所+.csv")
cc=df.loc[df["水表名"]=="养殖馆附房二楼厕所+"]
cc.to_csv("养殖馆附房二楼厕所+.csv")
cc=df.loc[df["水表名"]=="养殖馆附房一楼厕所+"]
cc.to_csv("养殖馆附房一楼厕所+.csv")
cc=df.loc[df["水表名"]=="养殖馆+"]
cc.to_csv("养殖馆+.csv")
cc=df.loc[df["水表名"]=="养殖队+"]
cc.to_csv("养殖队+.csv")
cc=df.loc[df["水表名"]=="XXX教学大楼总表"]
cc.to_csv("XXX教学大楼总表.csv")
cc=df.loc[df["水表名"]=="XXX中心大楼泵房"]
cc.to_csv("XXX中心大楼泵房.csv")
cc=df.loc[df["水表名"]=="XXX东大楼"]
cc.to_csv("XXX东大楼.csv")
cc=df.loc[df["水表名"]=="XXXM馆"]
cc.to_csv("XXXM馆.csv")
cc=df.loc[df["水表名"]=="XXX植物园"]
cc.to_csv("XXX植物园.csv")
cc=df.loc[df["水表名"]=="高配房+"]
cc.to_csv("高配房+.csv")
cc=df.loc[df["水表名"]=="养鱼组临工宿舍+"]
cc.to_csv("养鱼组临工宿舍+.csv")
cc=df.loc[df["水表名"]=="养鱼组厕所+"]
cc.to_csv("养鱼组厕所+.csv")
cc=df.loc[df["水表名"]=="农业试验站大棚+"]
cc.to_csv("农业试验站大棚+.csv")
cc=df.loc[df["水表名"]=="物业"]
cc.to_csv("物业.csv")
cc=df.loc[df["水表名"]=="体育馆网球场值班室"]
cc.to_csv("体育馆网球场值班室.csv")
cc=df.loc[df["水表名"]=="XXXS宾馆"]
cc.to_csv("XXXS宾馆.csv")
cc=df.loc[df["水表名"]=="东大门传达室+"]
cc.to_csv("东大门传达室+.csv")
df.groupby("水表名").describe()
cc5=df.groupby("水表名").describe()
cc5.to_csv("描述")
cc5.to_csv("描述.csv")
df["cha"]=df["当前读数"]-df["上次读数"]
df["cha"]=(df["当前读数"]-df["上次读数"]-df["用量"])
= df["采集时间"].strftime("%Y-%m-%d %H:%M:%S")
dc = df["采集时间"].strftime("%Y-%m-%d %H:%M:%S")
dc =pd.to_datetime( df["采集时间"] )
dc.day()
dc =pd.to_datetime( df["采集时间"], format='%Y%m%d' )
dc =pd.to_datetime( df["采集时间"], format='%Y%m%d%H%M%S' )
dc =pd.to_datetime( df["采集时间"], format='%Y/%m/%d/%H%M%S' )
cc6=df.groupby("水表名").sum()
cc6.to_csv("总用水.csv")
cc6=cc1.groupby("水表名").sum()

cc6.to_csv("总用水1.csv")
cc6=cc2.groupby("水表名").sum()

cc6.to_csv("总用水2.csv")
cc6=cc3.groupby("水表名").sum()

cc6.to_csv("总用水3.csv")
cc6=cc4.groupby("水表名").sum()

cc6.to_csv("总用水4.csv")
cc=df.loc[df["水表名"]=="校医院南+"]
bb=df.loc[df["水表名"]=="XXX校医院"]
aa=df.loc[df["水表名"]=="车队+"]
ff=pd.merge(cc,bb,aa,on="采集时间")
ff=pd.merge(cc,bb,on="采集时间")
gg=pd.merge(ff,aa,on="采集时间")
gg.to_csv("gg.csv")
gg["差"]=gg["用量_x"]-gg["用量_y"]-gg["用量"]
gg.loc[gg["差"]<0]
len(gg.loc[gg["差"]<0])
len(gg["差"].loc[gg["差"]<0])
gg["差"]=gg["用量_x"]-gg["用量_y"]
len(gg["差"].loc[gg["差"]<0])
cc=df.loc[df["水表名"]=="XXX成教院XXX分院"]
bb=df.loc[df["水表名"]=="离退休活动室"]
aa=df.loc[df["水表名"]=="XXXL馆"]
dd=df.loc[df["水表名"]=="XXXL楼"]
ff=df.loc[df["水表名"]=="XXXS馆"]
gg=df.loc[df["水表名"]=="XXXK"]
hh=df.loc[df["水表名"]=="XXXK酒店"]
jj=df.loc[df["水表名"]=="XXX体育馆"]
kk=df.loc[df["水表名"]=="XXX干训楼"]
cc=df.loc[df["水表名"]=="XXX成教院XXX分院"]
bb=df.loc[df["水表名"]=="离退休活动室"]
aa=df.loc[df["水表名"]=="XXXL馆"]
dd=df.loc[df["水表名"]=="XXXL楼"]
ff=df.loc[df["水表名"]=="XXXS馆"]
gg=df.loc[df["水表名"]=="XXXK"]
hh=df.loc[df["水表名"]=="XXXK酒店"]
jj=df.loc[df["水表名"]=="XXX体育馆"]
kk=df.loc[df["水表名"]=="XXX干训楼"]
zz=df.loc[df["水表名"]=="XXX花圃+"]

aa=pd.merge(zz,aa,on="采集时间")
aa=pd.merge(bb,aa,on="采集时间")
aa=pd.merge(cc,aa,on="采集时间")
aa=pd.merge(dd,aa,on="采集时间")
aa=pd.merge(ff,aa,on="采集时间")
aa=pd.merge(gg,aa,on="采集时间")
aa=pd.merge(hh,aa,on="采集时间")
aa=pd.merge(jj,aa,on="采集时间")
aa=pd.merge(kk,aa,on="采集时间")
aa
cc=df.loc[df["水表名"]=="XXX成教院XXX分院"]
bb=df.loc[df["水表名"]=="离退休活动室"]
aa=df.loc[df["水表名"]=="XXXL馆"]

zz=df.loc[df["水表名"]=="XXX花圃+"]

aa=pd.merge(zz,aa,on="采集时间")
aa=pd.merge(bb,aa,on="采集时间")
aa=pd.merge(cc,aa,on="采集时间")
cc=df.loc[df["水表名"]=="XXX成教院XXX分院"]
aa=df.loc[df["水表名"]=="XXXL馆"]

zz=df.loc[df["水表名"]=="XXX花圃+"]

aa=pd.merge(zz,aa,on="采集时间")
aa["差"]=aa["用量_x"]-aa["用量_y"]
len(gg["差"].loc[aa["差"]<0])
aa["差"]=aa["用量_x"]-aa["用量_y"]
len(aa["差"].loc[aa["差"]<0])
bb=df.loc[df["水表名"]=="离退休活动室"]
aa=df.loc[df["水表名"]=="XXXL馆"]

zz=df.loc[df["水表名"]=="XXX花圃+"]

aa=pd.merge(zz,aa,on="采集时间")
aa=pd.merge(bb,aa,on="采集时间")
aa["差"]=aa["用量_x"]-aa["用量_y"]-aa["用量"]
len(aa["差"].loc[aa["差"]<0])
cc=df.loc[df["水表名"]=="XXX成教院XXX分院"]
bb=df.loc[df["水表名"]=="离退休活动室"]
aa=df.loc[df["水表名"]=="XXXL馆"]

zz=df.loc[df["水表名"]=="XXX花圃+"]

aa=pd.merge(zz,aa,on="采集时间")
aa=pd.merge(bb,aa,on="采集时间")
aa=pd.merge(cc,aa,on="采集时间")
aa.drop(['上次读数', '上次读数_x',"水表名","水表名_x"], axis=1)
aa.drop(['上次读数', '上次读数_x',"水表名","水表名_x"])
aa.drop([['上次读数', '上次读数_x',"水表名","水表名_x"]], axis=1)
aa.drop(columns=['上次读数', '上次读数_x',"水表名","水表名_x"], axis=1)
aa.drop(columns=['水表号_x'], axis=1)
aa=aa.drop(columns=['水表号_x',"水表名"], axis=1)
aa=aa.drop(columns=[('水表号_x',"水表名")], axis=1)
aa=aa.drop(columns=[['水表号_x',"水表名"]], axis=1)
aa=aa.drop(columns=['水表号_x',"水表名"])
aa=aa.drop(columns=["水表名"])
aa=aa.drop(columns=["水表名_x"])
aa=aa.drop(columns=["水表号_x"])
aa=aa.drop(columns=["水表号_y"])
aa=pd.merge(zz,aa,on="采集时间")
aa=[]

cc=df.loc[df["水表名"]=="XXX成教院XXX分院"]
bb=df.loc[df["水表名"]=="离退休活动室"]
aa=df.loc[df["水表名"]=="XXXL馆"]

zz=df.loc[df["水表名"]=="XXX花圃+"]

aa=pd.merge(zz,aa,on="采集时间")
cc=df.loc[df["水表名"]=="XXX成教院XXX分院"]
bb=df.loc[df["水表名"]=="离退休活动室"]
aa=df.loc[df["水表名"]=="XXXL馆"]

zz=df.loc[df["水表名"]=="XXX花圃+"]

aa=pd.merge(zz,aa,on="采集时间")
aa=pd.merge(aa,bb,on="采集时间")
aa=pd.merge(aa,cc,on="采集时间")
aa.columns
aa["差"]=aa["用量_x"]-aa["用量_y"]
len(aa["差"].loc[aa["差"]<0])
aa.columns

aa.to_csv("ss.csv")
len(df["cha"].loc[df["cha"]<>0])
len(df["cha"].loc[df["cha"]!=0])
len(df["cha"].loc[(df["cha"]=0.1)|(f["cha"]=-0.1))]
len(df["cha"].loc[df["cha"]=0.1])
len(df["cha"].loc[df["cha"]==0.1])
len(df["cha"].loc[df["cha"]==0.01])
df["cha"]
df["cha"]=(df["当前读数"]-df["上次读数"]-df["用量"])
df["cha"]
df.to_csv("aa.csv")
len(df["cha"].loc[df["cha"]==0.01])
len(df["cha"].loc[df["cha"]==-0.01])
len(df["cha"].loc[df["cha"]==0.01])
len(df["cha"].loc[df["cha"]!=0])
len(df["cha"].loc[df["cha"]>0])
len(df["cha"].loc[df["cha"]==0.01])
len(df["cha"].loc[(df["cha"]>0) & (df["cha"]<0.01)])
len(df["cha"].loc[(df["cha"]>0) & (df["cha"]<0.02)])
len(df["cha"].loc[(df["cha"]>0) & (df["cha"]<0.011)])
len(df["cha"].loc[(df["cha"]>0.01) & (df["cha"]<0.011)])
len(df["cha"].loc[(df["cha"]<-0.01) & (df["cha"]>-0.011)])
len(df["cha"].loc[(df["cha"]<-0.01) ])
len(df["cha"].loc[(df["cha"]>-0.01) & (df["cha"]<0)])
len(df["cha"].loc[df["cha"]==-0.01])
len(df["cha"].loc[(df["cha"]>-0.02) & (df["cha"]<0)])
len(df["cha"].loc[(df["cha"]<-0.01) & (df["cha"]>-0.11)])
len(df["cha"].loc[(df["cha"]<-0.01) & (df["cha"]>-0.2)])
len(df["cha"].loc[(df["cha"]>0)])
len(df["cha"].loc[(df["cha"]==0)])
df["ss"]=df["当前读数"]-df["上次读数"]
aa=df.loc[df["ss"]!=0]
df["ss"]=df["当前读数"]-df["上次读数"]
df["ss"]=df["ss"]-df["用量"]
aa=df.loc[df["ss"]!=0]
cc1=pd.read_csv("附件_一季度.csv",encoding="gbk")
cc1=pd.read_csv("附件_一季度.csv",encoding="gbk")
cc2=pd.read_csv("附件_二季度.csv",encoding="gbk")
cc3=pd.read_csv("附件_三季度.csv",encoding="gbk")
cc4=pd.read_csv("附件_四季度.csv",encoding="gbk")
df = pd.concat([cc1, cc2,cc3,cc4])
cc5=df.groupby("水表名").describe()
cc5=df.groupby("水表名")
cc5
cc6=cc4.groupby("水表名").sum()
bb=df.loc[df["水表名"]=="XXX校医院"]
df['水表名'].unique()
"ming"=df['水表名'].unique()
ming=df['水表名'].unique()
aa=df.loc[df["水表名"]==i]
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa[0]
    
    print(aa)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa[:,0]
    
    print(aa)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa[:,1]
    
    print(aa)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa(1)
    
    print(aa)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa.iloc[1]
    
    print(aa)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa.iloc[1]
    bb=aa.iloc[-1]
    cc=bb-aa
    print(cc)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa.iloc['当前读数']
    bb=aa.iloc[-1]
    dd=aa.iloc[1]
    cc=bb-dd
    print(cc)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa.loc['当前读数']
    bb=aa.iloc[-1]
    dd=aa.iloc[1]
    cc=bb-dd
    print(cc)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa.loc[['当前读数']]
    bb=aa.iloc[-1]
    dd=aa.iloc[1]
    cc=bb-dd
    print(cc)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa['当前读数']
    bb=aa.iloc[-1]
    dd=aa.iloc[1]
    cc=bb-dd
    print(cc)
ming
ming.T
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa['当前读数']
    bb=aa.iloc[-1]
    dd=aa.iloc[1]
    cc=bb-dd
    print(cc+i)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa['当前读数']
    bb=aa.iloc[-1]
    dd=aa.iloc[1]
    cc=bb-dd
    print(cc+"i")
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa['当前读数']
    bb=aa.iloc[-1]
    dd=aa.iloc[1]
    cc=bb-dd
    print(cc&"i")
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa['当前读数']
    bb=aa.iloc[-1]
    dd=aa.iloc[1]
    cc=bb-dd
    print(cc)
    print(i)
import pandas as pd 

import numpy as np
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
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa[['当前读数',"采集时间"]]
    bb=aa.iloc[-1]
    dd=aa.iloc[1]
    cc=bb-dd
    print(cc)
    print(i)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    aa=aa[['当前读数',"采集时间"]]
    plt.plot(aa["当前读数"],aa["采集时间"],color='red',linewidth=2.0,linestyle='--')
    plt.show()
import pandas as pd 

import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题

cc1=pd.read_csv("附件_一季度.csv",encoding="gbk")
cc2=pd.read_csv("附件_二季度.csv",encoding="gbk")
cc3=pd.read_csv("附件_三季度.csv",encoding="gbk")
cc4=pd.read_csv("附件_四季度.csv",encoding="gbk")
df = pd.concat([cc1, cc2,cc3,cc4])
ming=df['水表名'].unique()
ming[0]
ming[0:10]
for i in ming[0:10]:
    aa=df.loc[df["水表名"]==i]
    aa=aa[['当前读数',"采集时间"]]
    plt.plot(aa["当前读数"],aa["采集时间"],color='red',linewidth=2.0,linestyle='--')
    plt.show()
for i in ming[0:1]:
    aa=df.loc[df["水表名"]==i]
    aa=aa[['当前读数',"采集时间"]]
    plt.plot(aa["当前读数"],aa["采集时间"],color='red',linewidth=2.0,linestyle='--')
    plt.show()
for i in ming[0:1]:
    aa=df.loc[df["水表名"]==i]
    aa=aa[['当前读数',"采集时间"]]
    plt.plot(aa["当前读数"],aa["采集时间"],color='red',linewidth=2.0,linestyle='--')
    plt.show()
aa=df.loc[df["水表名"]=="司法鉴定中心"]
aa=aa[['当前读数',"采集时间"]]
plt.plot(aa["当前读数"],aa["采集时间"],color='red',linewidth=2.0,linestyle='--')
plt.show()

df
cc5=df.groupby("水表名").min()
cc5
cc5.to_csv("min.csv")
cc5=df.groupby(["水表名","当前读表"]).min()
cc5=df.groupby(["水表名","当前读数"]).min()
cc5=df.groupby("水表名")["当前读数"].min()
cc5
cc5=df.groupby("水表名")["当前读数"].min().unstack()
cc5=df.groupby("水表名")min()
cc5=df.groupby("水表名").min()
cc5=df.groupby("水表名")["当前读数"].min()
cc5=df.groupby("水表名")
cc5.min()
aa=df.loc[df["水表名"]=="XXX国际纳米研究所"]
aa
cc1=pd.read_csv("附件_一季度.csv",encoding="gbk")
cc2=pd.read_csv("附件_二季度.csv",encoding="gbk")
cc3=pd.read_csv("附件_三季度.csv",encoding="gbk")
cc4=pd.read_csv("附件_四季度.csv",encoding="gbk")

df = pd.concat([cc1, cc2,cc3,cc4])
aa=df.loc[df["水表名"]=="XXX国际纳米研究所"]
aa
aa=df.loc[df["水表名"]=="XXX国际纳米研究所"]
aa
cc5=df.groupby("水表名").describe()
cc5
aa=df.loc[df["水表名"]=="XXX国际纳米研究所\t"]
aa
aa.to_csv("XXX国际纳米研究所\t.csv")
aa.to_csv("XXX国际纳米研究所.csv")
aa=df.loc[df["水表名"]=="64397副表"]
df.describe()
df["水表号"].unique()
df["水表号"].unique().size()
len(df["水表号"].unique())
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题

cc1=pd.read_csv("附件_一季度.csv",encoding="gbk")
cc2=pd.read_csv("附件_二季度.csv",encoding="gbk")
cc3=pd.read_csv("附件_三季度.csv",encoding="gbk")
cc4=pd.read_csv("附件_四季度.csv",encoding="gbk")

df = pd.concat([cc1, cc2,cc3,cc4])
cc6=cc4.groupby("水表名").sum()
for i in ming:
    aa=df.loc[df["水表名"]==i]
    bb=aa["当前读数"].max()
    cc=aa["当前读数"].min()
aa=df.loc[df["水表名"]=="64397副表"]
bb=aa["当前读数"].max()
for i in ming:
    aa=df.loc[df["水表名"]==i]
    bb=aa["当前读数"].max()
    cc=aa["当前读数"].min()
    dd=bb-cc
    print(dd)
    print(i)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    bb=aa["当前读数"].max()
    cc=aa["当前读数"].min()
    ff=aa["当前读数"].mean()
    dd=(bb-cc)/ff
    print(dd)
    print(i)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    bb=aa["当前读数"].max()
    cc=aa["当前读数"].min()
    ff=aa["当前读数"].mean()
    dd=(bb-cc)/35039
    print(dd)
    print(i)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    bb=aa["当前读数"].max()
    cc=aa["当前读数"].min()
    ff=aa["当前读数"].mean()
    dd=(bb-cc)/35039
    print("dd"+i)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    bb=aa["当前读数"].max()
    cc=aa["当前读数"].min()
    ff=aa["当前读数"].mean()
    dd=(bb-cc)/35039
    print(dd+"i")
    print(i)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    bb=aa["当前读数"].max()
    cc=aa["当前读数"].min()
    ff=aa["当前读数"].mean()
    dd=(bb-cc)/35039
    print(dd ++ i)
    print(i)
for i in ming:
    aa=df.loc[df["水表名"]==i]
    bb=aa["当前读数"].max()
    cc=aa["当前读数"].min()
    ff=aa["当前读数"].mean()
    dd=(bb-cc)/35039
    print(dd)
    print(i)
cc5=df.groupby("水表名").describe()
cc5

cc5.to_csv("描述.csv")
aa=df.loc[df["水表名"]=="纳米楼厕所+"]
aa.to_csv("纳米楼厕所+.csv")
cc1=pd.read_csv("附件_一季度.csv",encoding="gbk")
cc2=pd.read_csv("附件_二季度.csv",encoding="gbk")
cc3=pd.read_csv("附件_三季度.csv",encoding="gbk")
cc4=pd.read_csv("附件_四季度.csv",encoding="gbk")

df = pd.concat([cc1, cc2,cc3,cc4])
ming=df['水表名'].unique()
aa=df.loc[df["水表名"]=="XXX成教院XXX分院"]
bb=aa["当前读数"]
bb.iloc(0,1)

aa=df.loc[df["水表名"]=="XXX成教院XXX分院"]
bb=aa["当前读数"]
bb.iloc[0,1]

aa=df.loc[df["水表名"]=="XXX成教院XXX分院"]
bb=aa["当前读数"]
bb.iloc[0]

aa=df.loc[df["水表名"]=="XXX成教院XXX分院"]
bb=aa["当前读数"]
bb.iloc[-1]

ming=df['水表名'].unique()

for i in ming:
    aa=df.loc[df["水表名"]==i]
    bb=aa["当前读数"]
    
    dd=bb.iloc[-1]-bb.iloc[0]
    print(dd)
    print(i)
cc1["cha"]=cc1["当前读数"]-cc1["上次读数"]
cc2["cha"]=cc2["当前读数"]-cc2["上次读数"]
cc3["cha"]=cc3["当前读数"]-cc3["上次读数"]
cc4["cha"]=cc4["当前读数"]-cc4["上次读数"]
df = pd.concat([cc1, cc2,cc3,cc4])
cc6=cc4.groupby("水表名").sum()
cc6
cc6.to_csv("chazhi.csv")
cc99=pd.read_csv("水表层级.csv",encoding="gbk")
cc7=pd.merge(cc99,cc6,on="水表名")
cc7.to_csv("buzhiming.csv")
cc6=df.groupby("水表名").sum()
cc7=pd.merge(cc99,cc6,on="水表名")
cc7.to_csv("buzhiming.csv")
cc=df.loc[df["水表名"]=="消防"]
cc
df["RecTm"]=pd.to_datetime(df['采集时间'])
df = pd.concat([cc1, cc2,cc3,cc4])
df["用量"]-df["cha"]
cc10=df["用量"]-df["cha"]
len(cc10!=0])
len(cc10!=0)
cc10!=0
df["cha"]=(df["当前读数"]-df["上次读数"]-df["用量"])
len(df["cha"].loc[(df["cha"]>0.00001) & (df["cha"]<0.01)])
len(df["cha"].loc[(df["cha"]>0.00001))
len(df["cha"].loc[(df["cha"]>0.00001)])
len(df["cha"].loc[(df["cha"]<-0.00001)])
cc1=pd.read_excel("一季度.xlsx")
cc2=pd.read_excel("二季度.xlsx")
cc3=pd.read_excel("三季度.xlsx")
cc4=pd.read_excel("四季度.xlsx")
cc1=pd.read_excel("一季度.xlsx")
cc1=pd.read_excel("一季度.xlsx")

cc2=pd.read_excel("二季度.xlsx")
cc3=pd.read_excel("三季度.xlsx")
cc4=pd.read_excel("四季度.xlsx")

clean
clc
cc1=pd.read_csv("一季度.csv",encoding="gbk")
cc2=pd.read_csv("二季度.csv",encoding="gbk")
cc3=pd.read_csv("三季度.csv",encoding="gbk")
cc4=pd.read_csv("四季度.csv",encoding="gbk")
df = pd.concat([cc1, cc2,cc3,cc4])
len(df["时间"].loc[(df["时间"]>2)&(df["时间"]<5)])
cc11=df["时间"].loc[(df["时间"]>2)&(df["时间"]<5)]
cc11=df.loc[(df["时间"]>2)&(df["时间"]<5)]
cc11.groupby("水表名")["用量"].sum()
cc12=cc11.groupby("水表名")["用量"].sum()
cc12.to_csv("半夜出水.csv")
cc11=df.loc[(df["时间"]>6)&(df["时间"]<18)]
cc1
cc11
cc11.to_csv("白天.csv")
cc11
from sklearn.cluster import DBSCAN

X = cc11["用量"]
# 设置半径为10，最小样本量为2，建模
db = DBSCAN(eps=10, min_samples=2).fit(X)
X = cc11["用量"].reshape(-1,1)# 设置半径为10，最小样本量为2，建模
db = DBSCAN(eps=10, min_samples=2).fit(X)
X = np.array(cc11["用量"]).reshape(-1,1)# 设置半径为10，最小样本量为2，建模
db = DBSCAN(eps=10, min_samples=2).fit(X)
cc1=pd.read_csv("一季度.csv",encoding="gbk")
cc2=pd.read_csv("二季度.csv",encoding="gbk")
cc3=pd.read_csv("三季度.csv",encoding="gbk")
cc4=pd.read_csv("四季度.csv",encoding="gbk")
df = pd.concat([cc1, cc2,cc3,cc4])
cc5=df.groupby("水表名")["采集时间"].sum()
cc5
cc5=df.groupby("水表名").sum()
cc5
cc5=df.groupby(["水表名","采集时间"]).sum()
cc5
X = np.array(cc5["用量"]).reshape(-1,1)# 设置半径为10，最小样本量为2，建模
db = DBSCAN(eps=10, min_samples=2).fit(X)

labels = db.labels_ 
cc5['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果
cc5.sort_values('cluster_db')
from sklearn.cluster import DBSCAN
X = np.array(cc5["用量"]).reshape(-1,1)# 设置半径为10，最小样本量为2，建模
db = DBSCAN(eps=10, min_samples=2).fit(X)

labels = db.labels_ 
cc5['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果
cc5.sort_values('cluster_db')
cc6=cc5.sort_values('cluster_db')
cc6.to_cav("聚类结果")
cc6.to_csv("聚类结果")
cc6.to_csv("聚类结果.csv")
print(cc5.groupby('cluster_db').mean())
print(pd.scatter_matrix(X, c=colors[cc5.cluster_db], figsize=(10,10), s=100))
from sklearn.cluster import DBSCAN
cc11=df.loc[(df["时间"]>6)&(df["时间"]<18)]

X = np.array(cc5["用量"]).reshape(-1,1)# 设置半径为10，最小样本量为2，建模
db = DBSCAN(eps=15, min_samples=3).fit(X)

labels = db.labels_ 
cc5['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果
cc5.sort_values('cluster_db')
print(cc5.groupby('cluster_db').mean())
cc6=cc5.sort_values('cluster_db')
cc6
from random import sample, random, choice, randint
from math import ceil, log


class Node(object):
    def __init__(self, size):
        """Node class to build tree leaves
        
        Keyword Arguments:
            size {int} -- Node size (default: {None})
        """
        
        # Node size
        self.size = size
        # Feature to split
        self.split_feature = None
        # Split point
        self.split_point = None
        # Left child node
        self.left = None
        # Right child node
        self.right = None



class IsolationTree(object):
    def __init__(self, X, n_samples, max_depth):
        """Isolation Tree class
        
        Arguments:
            X {list} -- 2d list with int or float
            n_samples {int} -- Subsample size
            max_depth {int} -- Maximum height of isolation tree
        """
        self.height = 0
        # In case of n_samples is greater than n
        n = len(X)
        if n_samples > n:
            n_samples = n
        # Root node
        self.root = Node(n_samples)
        # Build isolation tree
        self._build_tree(X, n_samples, max_depth)
    
    def _get_split(self, X, idx, split_feature):
        """Randomly choose a split point
        
        Arguments:
            X {list} -- 2d list object with int or float
            idx {list} -- 1d list object with int
            split_feature {int} -- Column index of X
        
        Returns:
            int -- split point
        """
        
        # The split point should be greater than min(X[feature])
        unique = set(map(lambda i: X[i][split_feature], idx))
        # Cannot split
        if len(unique) == 1:
            return None
        unique.remove(min(unique))
        x_min, x_max = min(unique), max(unique)
        # Caution: random() -> x in the interval [0, 1).
        return random() * (x_max - x_min) + x_min
    
    def _build_tree(self, X, n_samples, max_depth):
        """The current node data space is divided into 2 sub space: less than the
        split point in the specified dimension on the left child of the current node,
        put greater than or equal to split point data on the current node's right child.
        Recursively construct new child nodes until the data cannot be splitted in the
        child nodes or the child nodes have reached the max_depth.
        
        Arguments:
            X {list} -- 2d list object with int or float
            n_samples {int} -- Subsample size
            max_depth {int} -- Maximum depth of IsolationTree
        """
        
        # Dataset shape
        m = len(X[0])
        n = len(X)
        # Randomly selected sample points into the root node of the tree
        idx = sample(range(n), n_samples)
        # Depth, Node and idx
        que = [[0, self.root, idx]]
        # BFS
        while que and que[0][0] <= max_depth:
            depth, nd, idx = que.pop(0)
            # Stop split if X cannot be splitted
            nd.split_feature = choice(range(m))
            nd.split_point = self._get_split(X, idx, nd.split_feature)
            if nd.split_point is None:
                continue
            # Split
            idx_left = []
            idx_right = []
            while idx:
                i = idx.pop()
                xi = X[i][nd.split_feature]
                if xi < nd.split_point:
                    idx_left.append(i)
                else:
                    idx_right.append(i)
            # Generate left and right child
            nd.left = Node(len(idx_left))
            nd.right = Node(len(idx_right))
            # Put the left and child into the que and depth plus one
            que.append([depth+1, nd.left, idx_left])
            que.append([depth+1, nd.right, idx_right])
        # Update the height of IsolationTree
        self.height = depth
    
    def _predict(self, xi):
        """Auxiliary function of predict.
        
        Arguments:
            xi {list} -- 1D list with int or float
        
        Returns:
            int -- the depth of the node which the xi belongs to
        """
        
        # Search xi from the IsolationTree until xi is at an leafnode
        nd = self.root
        depth = 0
        while nd.left and nd.right:
            if xi[nd.split_feature] < nd.split_point:
                nd = nd.left
            else:
                nd = nd.right
            depth += 1
        return depth, nd.size



class IsolationForest(object):
    def __init__(self):
        """IsolationForest, randomly build some IsolationTree instance,
        and the average score of each IsolationTree
        
        
        Attributes:
        trees {list} -- 1d list with IsolationTree objects
        ajustment {float}
        """
        
        self.trees = None
        self.adjustment = None  # TBC
    
    def fit(self, X, n_samples=100, max_depth=10, n_trees=256):
        """Build IsolationForest with dataset X
        
        Arguments:
            X {list} -- 2d list with int or float
        
        Keyword Arguments:
            n_samples {int} -- According to paper, set number of samples to 256 (default: {256})
            max_depth {int} -- Tree height limit (default: {10})
            n_trees {int} --  According to paper, set number of trees to 100 (default: {100})
        """
        
        self.adjustment = self._get_adjustment(n_samples)
        self.trees = [IsolationTree(X, n_samples, max_depth)
                      for _ in range(n_trees)]
    
    def _get_adjustment(self, node_size):
        """Calculate adjustment according to the formula in the paper.
        
        Arguments:
            node_size {int} -- Number of leaf nodes
        
        Returns:
            float -- ajustment
        """
        
        if node_size > 2:
            i = node_size - 1
            ret = 2 * (log(i) + 0.5772156649) - 2 * i / node_size
        elif node_size == 2:
            ret = 1
        else:
            ret = 0
        return ret
    
    def _predict(self, xi):
        """Auxiliary function of predict.
        
        Arguments:
            xi {list} -- 1d list object with int or float
        
        Returns:
            list -- 1d list object with float
        """
        
        # Calculate average score of xi at each tree
        score = 0
        n_trees = len(self.trees)
        for tree in self.trees:
            depth, node_size = tree._predict(xi)
            score += (depth + self._get_adjustment(node_size))
        score = score / n_trees
        # Scale
        return 2 ** -(score / self.adjustment)
    
    def predict(self, X):
        """Get the prediction of y.
        
        Arguments:
            X {list} -- 2d list object with int or float
        
        Returns:
            list -- 1d list object with float
        """
        
        return [self._predict(xi) for xi in X]




def main():
    print("Comparing average score of X and outlier's score...")
    # Generate a dataset randomly
    n = 100
    X = cc5["用水量"]
    # Add outliers
    X.append([10]*5)
    # Train model
    clf = IsolationForest()
    clf.fit(X, n_samples=500)
    # Show result
    print("Average score is %.2f" % (sum(clf.predict(X)) / len(X)))
    print("Outlier's score is %.2f" % clf._predict(X[-1]))



if __name__ == "__main__":
    main()
from random import sample, random, choice, randint
from math import ceil, log


class Node(object):
    def __init__(self, size):
        """Node class to build tree leaves
        
        Keyword Arguments:
            size {int} -- Node size (default: {None})
        """
        
        # Node size
        self.size = size
        # Feature to split
        self.split_feature = None
        # Split point
        self.split_point = None
        # Left child node
        self.left = None
        # Right child node
        self.right = None



class IsolationTree(object):
    def __init__(self, X, n_samples, max_depth):
        """Isolation Tree class
        
        Arguments:
            X {list} -- 2d list with int or float
            n_samples {int} -- Subsample size
            max_depth {int} -- Maximum height of isolation tree
        """
        self.height = 0
        # In case of n_samples is greater than n
        n = len(X)
        if n_samples > n:
            n_samples = n
        # Root node
        self.root = Node(n_samples)
        # Build isolation tree
        self._build_tree(X, n_samples, max_depth)
    
    def _get_split(self, X, idx, split_feature):
        """Randomly choose a split point
        
        Arguments:
            X {list} -- 2d list object with int or float
            idx {list} -- 1d list object with int
            split_feature {int} -- Column index of X
        
        Returns:
            int -- split point
        """
        
        # The split point should be greater than min(X[feature])
        unique = set(map(lambda i: X[i][split_feature], idx))
        # Cannot split
        if len(unique) == 1:
            return None
        unique.remove(min(unique))
        x_min, x_max = min(unique), max(unique)
        # Caution: random() -> x in the interval [0, 1).
        return random() * (x_max - x_min) + x_min
    
    def _build_tree(self, X, n_samples, max_depth):
        """The current node data space is divided into 2 sub space: less than the
        split point in the specified dimension on the left child of the current node,
        put greater than or equal to split point data on the current node's right child.
        Recursively construct new child nodes until the data cannot be splitted in the
        child nodes or the child nodes have reached the max_depth.
        
        Arguments:
            X {list} -- 2d list object with int or float
            n_samples {int} -- Subsample size
            max_depth {int} -- Maximum depth of IsolationTree
        """
        
        # Dataset shape
        m = len(X[0])
        n = len(X)
        # Randomly selected sample points into the root node of the tree
        idx = sample(range(n), n_samples)
        # Depth, Node and idx
        que = [[0, self.root, idx]]
        # BFS
        while que and que[0][0] <= max_depth:
            depth, nd, idx = que.pop(0)
            # Stop split if X cannot be splitted
            nd.split_feature = choice(range(m))
            nd.split_point = self._get_split(X, idx, nd.split_feature)
            if nd.split_point is None:
                continue
            # Split
            idx_left = []
            idx_right = []
            while idx:
                i = idx.pop()
                xi = X[i][nd.split_feature]
                if xi < nd.split_point:
                    idx_left.append(i)
                else:
                    idx_right.append(i)
            # Generate left and right child
            nd.left = Node(len(idx_left))
            nd.right = Node(len(idx_right))
            # Put the left and child into the que and depth plus one
            que.append([depth+1, nd.left, idx_left])
            que.append([depth+1, nd.right, idx_right])
        # Update the height of IsolationTree
        self.height = depth
    
    def _predict(self, xi):
        """Auxiliary function of predict.
        
        Arguments:
            xi {list} -- 1D list with int or float
        
        Returns:
            int -- the depth of the node which the xi belongs to
        """
        
        # Search xi from the IsolationTree until xi is at an leafnode
        nd = self.root
        depth = 0
        while nd.left and nd.right:
            if xi[nd.split_feature] < nd.split_point:
                nd = nd.left
            else:
                nd = nd.right
            depth += 1
        return depth, nd.size



class IsolationForest(object):
    def __init__(self):
        """IsolationForest, randomly build some IsolationTree instance,
        and the average score of each IsolationTree
        
        
        Attributes:
        trees {list} -- 1d list with IsolationTree objects
        ajustment {float}
        """
        
        self.trees = None
        self.adjustment = None  # TBC
    
    def fit(self, X, n_samples=100, max_depth=10, n_trees=256):
        """Build IsolationForest with dataset X
        
        Arguments:
            X {list} -- 2d list with int or float
        
        Keyword Arguments:
            n_samples {int} -- According to paper, set number of samples to 256 (default: {256})
            max_depth {int} -- Tree height limit (default: {10})
            n_trees {int} --  According to paper, set number of trees to 100 (default: {100})
        """
        
        self.adjustment = self._get_adjustment(n_samples)
        self.trees = [IsolationTree(X, n_samples, max_depth)
                      for _ in range(n_trees)]
    
    def _get_adjustment(self, node_size):
        """Calculate adjustment according to the formula in the paper.
        
        Arguments:
            node_size {int} -- Number of leaf nodes
        
        Returns:
            float -- ajustment
        """
        
        if node_size > 2:
            i = node_size - 1
            ret = 2 * (log(i) + 0.5772156649) - 2 * i / node_size
        elif node_size == 2:
            ret = 1
        else:
            ret = 0
        return ret
    
    def _predict(self, xi):
        """Auxiliary function of predict.
        
        Arguments:
            xi {list} -- 1d list object with int or float
        
        Returns:
            list -- 1d list object with float
        """
        
        # Calculate average score of xi at each tree
        score = 0
        n_trees = len(self.trees)
        for tree in self.trees:
            depth, node_size = tree._predict(xi)
            score += (depth + self._get_adjustment(node_size))
        score = score / n_trees
        # Scale
        return 2 ** -(score / self.adjustment)
    
    def predict(self, X):
        """Get the prediction of y.
        
        Arguments:
            X {list} -- 2d list object with int or float
        
        Returns:
            list -- 1d list object with float
        """
        
        return [self._predict(xi) for xi in X]




def main():
    print("Comparing average score of X and outlier's score...")
    # Generate a dataset randomly
    n = 100
    X = cc5["用量"]
    # Add outliers
    X.append([10]*5)
    # Train model
    clf = IsolationForest()
    clf.fit(X, n_samples=500)
    # Show result
    print("Average score is %.2f" % (sum(clf.predict(X)) / len(X)))
    print("Outlier's score is %.2f" % clf._predict(X[-1]))



if __name__ == "__main__":
    main()
from random import sample, random, choice, randint
from math import ceil, log


class Node(object):
    def __init__(self, size):
        """Node class to build tree leaves
        
        Keyword Arguments:
            size {int} -- Node size (default: {None})
        """
        
        # Node size
        self.size = size
        # Feature to split
        self.split_feature = None
        # Split point
        self.split_point = None
        # Left child node
        self.left = None
        # Right child node
        self.right = None



class IsolationTree(object):
    def __init__(self, X, n_samples, max_depth):
        """Isolation Tree class
        
        Arguments:
            X {list} -- 2d list with int or float
            n_samples {int} -- Subsample size
            max_depth {int} -- Maximum height of isolation tree
        """
        self.height = 0
        # In case of n_samples is greater than n
        n = len(X)
        if n_samples > n:
            n_samples = n
        # Root node
        self.root = Node(n_samples)
        # Build isolation tree
        self._build_tree(X, n_samples, max_depth)
    
    def _get_split(self, X, idx, split_feature):
        """Randomly choose a split point
        
        Arguments:
            X {list} -- 2d list object with int or float
            idx {list} -- 1d list object with int
            split_feature {int} -- Column index of X
        
        Returns:
            int -- split point
        """
        
        # The split point should be greater than min(X[feature])
        unique = set(map(lambda i: X[i][split_feature], idx))
        # Cannot split
        if len(unique) == 1:
            return None
        unique.remove(min(unique))
        x_min, x_max = min(unique), max(unique)
        # Caution: random() -> x in the interval [0, 1).
        return random() * (x_max - x_min) + x_min
    
    def _build_tree(self, X, n_samples, max_depth):
        """The current node data space is divided into 2 sub space: less than the
        split point in the specified dimension on the left child of the current node,
        put greater than or equal to split point data on the current node's right child.
        Recursively construct new child nodes until the data cannot be splitted in the
        child nodes or the child nodes have reached the max_depth.
        
        Arguments:
            X {list} -- 2d list object with int or float
            n_samples {int} -- Subsample size
            max_depth {int} -- Maximum depth of IsolationTree
        """
        
        # Dataset shape
        m = len(X[0])
        n = len(X)
        # Randomly selected sample points into the root node of the tree
        idx = sample(range(n), n_samples)
        # Depth, Node and idx
        que = [[0, self.root, idx]]
        # BFS
        while que and que[0][0] <= max_depth:
            depth, nd, idx = que.pop(0)
            # Stop split if X cannot be splitted
            nd.split_feature = choice(range(m))
            nd.split_point = self._get_split(X, idx, nd.split_feature)
            if nd.split_point is None:
                continue
            # Split
            idx_left = []
            idx_right = []
            while idx:
                i = idx.pop()
                xi = X[i][nd.split_feature]
                if xi < nd.split_point:
                    idx_left.append(i)
                else:
                    idx_right.append(i)
            # Generate left and right child
            nd.left = Node(len(idx_left))
            nd.right = Node(len(idx_right))
            # Put the left and child into the que and depth plus one
            que.append([depth+1, nd.left, idx_left])
            que.append([depth+1, nd.right, idx_right])
        # Update the height of IsolationTree
        self.height = depth
    
    def _predict(self, xi):
        """Auxiliary function of predict.
        
        Arguments:
            xi {list} -- 1D list with int or float
        
        Returns:
            int -- the depth of the node which the xi belongs to
        """
        
        # Search xi from the IsolationTree until xi is at an leafnode
        nd = self.root
        depth = 0
        while nd.left and nd.right:
            if xi[nd.split_feature] < nd.split_point:
                nd = nd.left
            else:
                nd = nd.right
            depth += 1
        return depth, nd.size



class IsolationForest(object):
    def __init__(self):
        """IsolationForest, randomly build some IsolationTree instance,
        and the average score of each IsolationTree
        
        
        Attributes:
        trees {list} -- 1d list with IsolationTree objects
        ajustment {float}
        """
        
        self.trees = None
        self.adjustment = None  # TBC
    
    def fit(self, X, n_samples=100, max_depth=10, n_trees=256):
        """Build IsolationForest with dataset X
        
        Arguments:
            X {list} -- 2d list with int or float
        
        Keyword Arguments:
            n_samples {int} -- According to paper, set number of samples to 256 (default: {256})
            max_depth {int} -- Tree height limit (default: {10})
            n_trees {int} --  According to paper, set number of trees to 100 (default: {100})
        """
        
        self.adjustment = self._get_adjustment(n_samples)
        self.trees = [IsolationTree(X, n_samples, max_depth)
                      for _ in range(n_trees)]
    
    def _get_adjustment(self, node_size):
        """Calculate adjustment according to the formula in the paper.
        
        Arguments:
            node_size {int} -- Number of leaf nodes
        
        Returns:
            float -- ajustment
        """
        
        if node_size > 2:
            i = node_size - 1
            ret = 2 * (log(i) + 0.5772156649) - 2 * i / node_size
        elif node_size == 2:
            ret = 1
        else:
            ret = 0
        return ret
    
    def _predict(self, xi):
        """Auxiliary function of predict.
        
        Arguments:
            xi {list} -- 1d list object with int or float
        
        Returns:
            list -- 1d list object with float
        """
        
        # Calculate average score of xi at each tree
        score = 0
        n_trees = len(self.trees)
        for tree in self.trees:
            depth, node_size = tree._predict(xi)
            score += (depth + self._get_adjustment(node_size))
        score = score / n_trees
        # Scale
        return 2 ** -(score / self.adjustment)
    
    def predict(self, X):
        """Get the prediction of y.
        
        Arguments:
            X {list} -- 2d list object with int or float
        
        Returns:
            list -- 1d list object with float
        """
        
        return [self._predict(xi) for xi in X]




def main():
    print("Comparing average score of X and outlier's score...")
    # Generate a dataset randomly
    n = 100
    X = cc5["用量"]
    # Add outliers
    
    # Train model
    clf = IsolationForest()
    clf.fit(X, n_samples=500)
    # Show result
    print("Average score is %.2f" % (sum(clf.predict(X)) / len(X)))
    print("Outlier's score is %.2f" % clf._predict(X[-1]))



if __name__ == "__main__":
    main()
from sklearn.ensemble import IsolationForest
import numpy as np
rng = np.random.RandomState(42)


clf = IsolationForest(max_samples=100*2)
clf.fit(cc5["用量"])
from sklearn.ensemble import IsolationForest
import numpy as np
rng = np.random.RandomState(42)


clf = IsolationForest(max_samples=100*2)
X = np.array(cc5["用量"]).reshape(-1,1)# 设置半径为10，最小样本量为2，建模

clf.fit(X)
y_pred_train = clf.predict(X)
y_pred_train
print(y_pred_train)
y_pred_train.to_csv("y_pred_train.csv")
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50)) # 生成网络数据 https://www.cnblogs.com/lemonbit/p/7593898.html
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r) # 等高线

b1 = plt.scatter(X[:, 0], X[:, 1], c='white',
                 s=20, edgecolor='k')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()
from sklearn.ensemble import IsolationForest
import numpy as np
rng = np.random.RandomState(42)


clf = IsolationForest(max_samples=100*2)
X = np.array(cc5["用量"]).reshape(-1,1)# 设置半径为10，最小样本量为2，建模

clf.fit(X)
y_pred_train = clf.predict(X)


plt.title("IsolationForest")


b1 = plt.scatter(X[:, 0], X[:, 1], c='white',
                 s=20, edgecolor='k')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()
model_isof = IsolationForest(n_estimators=20)
# 计算有无异常的标签分布
outlier_label = model_isof.fit_predict(X)
outlier_pd = pd.DataFrame(outlier_label, columns=['outlier_label'])
data_merge = pd.concat((df, outlier_pd), axis=1)
outlier_pd = pd.DataFrame(outlier_label, columns=['outlier_label'])
outlier_pd = pd.DataFrame(outlier_label, columns=['outlier_label'])
cc5["outlier_pd"]=outlier_pd
data_merge = pd.concat((cc5, outlier_pd), axis=1)
outlier_pd = pd.DataFrame(outlier_label, columns=['outlier_label'])
data_merge = pd.concat((cc5, outlier_pd))
data_merge.to_csv("孤立森林.csv")
reset
import pandas as pd 

import numpy as np
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
cc1=pd.read_csv("一季度.csv",encoding="gbk")
cc2=pd.read_csv("二季度.csv",encoding="gbk")
cc3=pd.read_csv("三季度.csv",encoding="gbk")
cc4=pd.read_csv("四季度.csv",encoding="gbk")
df = pd.concat([cc1, cc2,cc3,cc4])

cc5=df.groupby(["水表名","采集时间"]).sum()
from sklearn.ensemble import IsolationForest
import numpy as np
rng = np.random.RandomState(42)


clf = IsolationForest(max_samples=100*2)
X = np.array(cc5["用量"]).reshape(-1,1)# 设置半径为10，最小样本量为2，建模
model_isof = IsolationForest(n_estimators=20)
outlier_label = model_isof.fit_predict(X)
outlier_pd = pd.DataFrame(outlier_label, columns=['outlier_label'])
data_merge = pd.concat((cc5, outlier_pd))
cc5["outlier_pd"]=outlier_pd
X
df["cha"]=(df["当前读数"]-df["上次读数"]
df["cha"]=(df["当前读数"]-df["上次读数"])
df.to_csv("cha.csv")
cc5=df.groupby(["水表名","采集时间"])["用量"].value_counts()
cc5
cc5.to_csv("频次.csv")
cc5.to_excel("频次.xls")
cc5.to_excel("pinci.xls")
cc5=df.groupby(["水表名","采集时间"])["用量"].value_counts().tolist()
cc5
cc5=df.groupby(["水表名","采集时间"])["用量"].value_counts()
cc5
cc6=pd.read_excel("频次.xlsx")
ming=cc6["表名"].unque()
df=df.loc[(df["时间"]<5)&(df["时间"]>2)]
cc5=df.groupby(["水表名","采集时间"])["用量"].value_counts()
cc5
cc5.to_excel("pinci.xls")
cc
cc5
cc5=df.groupby(["水表名","采集时间"])["用量"].value_counts()
cc5
cc6=cc5.groupby(["水表名","采集时间"]])["频次"].sum()
cc5.to_excel("pinci.xlsm")
cc6=pd.read_excel("pinci.xlsm")
cc7=cc6.groupby(["水表名","采集时间"]])["频次"].sum()
cc7=cc6.groupby(["水表名","采集时间"])["频次"].sum()
cc7=cc6.groupby(["水表名","采集时间"])["用量.1"].sum()
cc7.to_excel("ercipinci.xlsx")
b1=pd.read_csv("b1.csv",encoding="gbk")
b2=pd.read_csv("b2.csv",encoding="gbk")
b3=pd.read_csv("b3.csv",encoding="gbk")
b4=pd.read_csv("b4.csv",encoding="gbk")
year_sum=pd.read_csv("niansum.csv",encoding="gbk")

#数据匹配
b11=pd.merge(b1,year_sum)
b22=pd.merge(b2,year_sum)
b33=pd.merge(b3,year_sum)
b44=pd.merge(b4,year_sum)
for i in b1:
    b5=b22[(b22["b2"]>i)&(b22["b2"]<i*100+99)]
    print(b5)
b5=b22[(b22["b2"]>401)&(b22["b2"]<401*100+99)]
b5
for i in b1:
    b5=b22[(b22["b2"]>int(i))&(b22["b2"]<int(i)*100+99)]
    print(b5)
b1=pd.Series(b1.values)
b1=pd.Series(b1)
b1=pd.Series(b1,index=df.index)
range(len(b1))
for i in b1:
    print(i)
    
for i in b1.values:
    print(i)
    
for i in b1.values:
    b5=b22[(b22["b2"]>int(i))&(b22["b2"]<int(i)*100+99)]
    print(b5)
for i in b1.values:
    b5=b22[(b22["b2"]>int(i))&(b22["b2"]<int(i)*100+99)]
    
    print(b5["cha"].sum())
for i in b1.values:
    b5=b22[(b22["b2"]>int(i))&(b22["b2"]<int(i)*100+99)]
    print(b5["cha"].sum())
    b1["下级和"]=b5["cha"].sum()
for i in b1.values:
    b5=[]
    b5=b22[(b22["b2"]>int(i))&(b22["b2"]<int(i)*100+99)]
    b1["下级和"]=b5["cha"].sum()
    print(b5["cha"].sum())
for i in b1.values:
    b5=b22[(b22["b2"]>int(i))&(b22["b2"]<int(i)*100+99)]
    print(b5)
    
reset
import pandas as pd 

import numpy as np
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
b1=pd.read_csv("b1.csv",encoding="gbk")
b2=pd.read_csv("b2.csv",encoding="gbk")
b3=pd.read_csv("b3.csv",encoding="gbk")
b4=pd.read_csv("b4.csv",encoding="gbk")
year_sum=pd.read_csv("niansum.csv",encoding="gbk")

#数据匹配
b11=pd.merge(b1,year_sum)
b22=pd.merge(b2,year_sum)
b33=pd.merge(b3,year_sum)
b44=pd.merge(b4,year_sum)
for i in b1.values:
    
    b5=b22[(b22["b2"]>int(i))&(b22["b2"]<int(i)*100+99)]
    print(b5["cha"].sum())
for i in b1.values:
    b5=b22[(b22["b2"]>int(i))&(b22["b2"]<int(i)*100+99)]
    print(b5)
    
for i in b1.values:
    
    b5=b22[(b22["b2"]>int(i)*100)&(b22["b2"]<int(i)*100+99)]
    print(b5)
b5=b22[(b22["b2"]>401*100)&(b22["b2"]<401*100+99)]
for i in b1.values:
    
    b5=b22[(b22["b2"]>int(i)*100)&(b22["b2"]<int(i)*100+99)]
    print(b5)
for i in b1.values:
    
    b5=b22[(b22["b2"]>int(i)*100)&(b22["b2"]<int(i)*100+99)]
    print(b5["cha"].sum())
import pandas as pd 

import numpy as np
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
cc7=cc6.groupby(["水表名","采集时间"]])["频次"].sum()
reset
cc6=pd.read_excel("pinci.xlsm")
"""
import pandas as pd 

import numpy as np
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
import pandas as pd 

import numpy as np
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
cc6=pd.read_excel("pinci.xlsm")
cc7=cc6.groupby(["水表名","采集时间"]])["用量.1"].sum()
cc7=cc6.groupby(["水表名","采集时间"])["用量.1"].sum()
reset
cc6=pd.read_excel("pinci.xlsm")
cc6=cc6[["水表名","采集时间","用量","用量.1"]]
cc6=pd.read_excel("ercipinci.xlsx")
len(cc6)
res = set()

for i in range(len(cc6)):
    for j in range(i+1,len(cc6)):
        res.add(data[j] - data[i])
res=[]

for i in range(len(cc6)):
    for j in range(i+1,len(cc6)):
        res.add(data[j] - data[i])
res=[]

for i in range(len(cc6)):
    for j in range(i+1,len(cc6)):
        res.append(data[j] - data[i])
l=len(cc6)
for i in range(l):
    for j in range(i+1,l):
        print(cc6[j] - cc6[i])
l=len(cc6)
for i in range(l):
    for j in range(i+1,l):
        print(cc6[j]-cc6[i])
cc6[0]
cc6[1]
cc6[0:1]
l=len(cc6)
for i in range(l):
    for j in range(i+1,l):
        print(cc6[i:j]-cc6[j:j+1])
import datetime
time=cc6["采集时间"]
startTime= time.strftime("%Y-%m-%d %H:%M:%s", time.localtime())
endTime= time.strftime("%Y-%m-%d %H:%M:%s", time.localtime())
startTime= datetime.datetime.strptime(startTime,"%Y-%m-%d %H:%M:%S")
endTime= datetime.datetime.strptime(endTime,"%Y-%m-%d %H:%M:%S")
# 相减得到秒数
seconds = (endTime- startTime).seconds
hours=(endTime- startTime).hours
day=(endTime- startTime).day
cc6["采集时间"]=pd.to_datetime(cc6["采集时间"])
import datetime
time=cc6["采集时间"]
startTime= time.strftime("%Y-%m-%d %H:%M:%s", time.localtime())
endTime= time.strftime("%Y-%m-%d %H:%M:%s", time.localtime())
startTime= datetime.datetime.strptime(startTime,"%Y-%m-%d %H:%M:%S")
endTime= datetime.datetime.strptime(endTime,"%Y-%m-%d %H:%M:%S")
# 相减得到秒数
seconds = (endTime- startTime).seconds
hours=(endTime- startTime).hours
day=(endTime- startTime).day
for i in range(l):
    print(cc6.iloc[1]  -cc6.iloc[1+1])
for i in range(l):
    cc7=cc6["采集时间"].iloc[1]  -cc6["采集时间"].iloc[1+1]
    print(cc7)
cc7=[]
for i in range(l):
    cc7.append(cc6["采集时间"].iloc[1]  -cc6["采集时间"].iloc[1+1])
    print(cc7)
cc7=[]
for i in range(l):
    cc7.append(cc6["采集时间"].iloc[1]  -cc6["采集时间"].iloc[1+1])

print(cc7)
cc7.to_csv("cc7.csv")
reset
import numpy as np
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
cc99=pd.read_csv("水表层级.csv",encoding="gbk")
cc1=pd.read_csv("附件_一季度.csv",encoding="gbk")
cc2=pd.read_csv("附件_二季度.csv",encoding="gbk")
cc3=pd.read_csv("附件_三季度.csv",encoding="gbk")
cc4=pd.read_csv("附件_四季度.csv",encoding="gbk")
cc1["cha"]=cc1["当前读数"]-cc1["上次读数"]
cc2["cha"]=cc2["当前读数"]-cc2["上次读数"]
cc3["cha"]=cc3["当前读数"]-cc3["上次读数"]
cc4["cha"]=cc4["当前读数"]-cc4["上次读数"]
df = pd.concat([cc1, cc2,cc3,cc4])
cc100=pd.read_csv("水表层级(1).csv",encoding="gbk")

## ---(Sat Sep 12 14:01:13 2020)---
from sklearn import  preprocessing
cc99=pd.read_csv("水表层级.csv",encoding="gbk")
cc1=pd.read_csv("附件_一季度.csv",encoding="gbk")
cc2=pd.read_csv("附件_二季度.csv",encoding="gbk")
cc3=pd.read_csv("附件_三季度.csv",encoding="gbk")
cc4=pd.read_csv("附件_四季度.csv",encoding="gbk")
cc1["cha"]=cc1["当前读数"]-cc1["上次读数"]
cc2["cha"]=cc2["当前读数"]-cc2["上次读数"]
cc3["cha"]=cc3["当前读数"]-cc3["上次读数"]
cc4["cha"]=cc4["当前读数"]-cc4["上次读数"]
df = pd.concat([cc1, cc2,cc3,cc4])
cc100=pd.read_csv("水表层级(1).csv",encoding="gbk")

"""
import pandas as pd 

import numpy as np
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
import pandas as pd 

import numpy as np
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
cc99=pd.read_csv("水表层级.csv",encoding="gbk")
cc1=pd.read_csv("附件_一季度.csv",encoding="gbk")
cc2=pd.read_csv("附件_二季度.csv",encoding="gbk")
cc3=pd.read_csv("附件_三季度.csv",encoding="gbk")
cc4=pd.read_csv("附件_四季度.csv",encoding="gbk")
cc1["cha"]=cc1["当前读数"]-cc1["上次读数"]
cc2["cha"]=cc2["当前读数"]-cc2["上次读数"]
cc3["cha"]=cc3["当前读数"]-cc3["上次读数"]
cc4["cha"]=cc4["当前读数"]-cc4["上次读数"]
df = pd.concat([cc1, cc2,cc3,cc4])
cc100=pd.read_csv("水表层级(1).csv",encoding="gbk")
cc111=pd.merge(df,cc100,on="水表名")
cc111=cc111[["水表名","水表号_x","采集时间","上次读数","当前读数","用量","cha","口径"]]
cc1=pd.read_excel("一季度.xlsx")
cc2=pd.read_excel("二季度.xlsx")
cc3=pd.read_excel("三季度.xlsx")
cc4=pd.read_excel("四季度.xlsx")
cc9=pd.read_excel("pinci.xlsm")
cc10=cc9.groupby(["水表号","采集时间"])["zong"].sum()
cc10=cc9.groupby(["水表名","采集时间"])["zong"].sum()
cc10.to_csv("cc10.csv")
cc10=cc9.groupby(["水表名","采集时间"])["用量.1"].sum()
cc10
cc10.to_csv("cc11.csv")
df = pd.concat([cc1, cc2,cc3,cc4])
cc111=pd.merge(df,cc100,on="水表名")
cc111=cc111[["水表名","水表号_x","采集时间","上次读数","当前读数","用量","cha","口径"]]
cc111=cc111[["水表名","水表号_x","采集时间","上次读数","当前读数","用量","口径"]]
cc110=cc111.groupby(["水表名","采集时间"])["用量"].sum()
cc110.to_csv("日用水合.csv")
cc112=pd.read_csv("日用水合.csv")
cc112=pd.read_csv("日用水合.csv",encoding="gbk")
cc113=pd.merge(cc112,cc100,on="水表名")
cc113=cc113[["水表名","采集时间","上次读数","当前读数","用量","口径"]]
cc113=cc113[["水表名","采集时间","用量","口径"]]
X =cc113[["用量","口径"]]
model = DBSCAN(eps=0.003, min_samples=1)
from sklearn.cluster import DBSCAN
yhat = model.fit_predict(X)
model = DBSCAN(eps=0.003, min_samples=1)
yhat = model.fit_predict(X)
labels = model.labels_
cc113['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果
model = DBSCAN(eps=10, min_samples=2)
yhat = model.fit_predict(X)
labels = model.labels_
cc113['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果
cc113.to_csv("白天.csv")
from sklearn.ensemble import IsolationForest
import numpy as np
X = np.array(X).reshape(-1,1)# 设置半径为10，最小样本量为2，建模

model_isof = IsolationForest(n_estimators=20)
outlier_label = model_isof.fit_predict(X)
outlier_pd = pd.DataFrame(outlier_label, columns=['outlier_label'])
data_merge = pd.concat((df, outlier_pd), axis=1)
data_merge = pd.concat((df, outlier_pd))
data_merge["outlier_label"].value_counts()
outlier_pd.to_csv("outlier_pd.csv")
X
cc13["outlier_pd"]=outlier_label
cc113["outlier_pd"]=outlier_label
X
X =cc113[["用量","口径"]]
model_isof = IsolationForest(n_estimators=20)
outlier_label = model_isof.fit_predict(X)
cc113["outlier_pd"]=outlier_label
outlier_label.value_counts()
cc113["outlier_pd"].value_counts()
cc113.to_csv("outlier_pd.csv")

## ---(Sat Sep 12 16:07:27 2020)---
import pandas as pd 

import numpy as np
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
cc1=pd.read_excel("一季度.xlsx")
cc2=pd.read_excel("二季度.xlsx")
cc3=pd.read_excel("三季度.xlsx")
cc4=pd.read_excel("四季度.xlsx")
df = pd.concat([cc1, cc2,cc3,cc4])
df.loc[["XXX花圃+"]]
df.loc["水表名"==["XXX花圃+"]]
df.loc["水表名"].isin("花圃")
df.loc[df["水表名"].isin("花圃")]
df["水表名"]=="XXX花圃+"
df.loc[df["水表名"]=="XXX花圃+"]
df.loc[df["水表名"]==[["XXX花圃+","XXX成教院XXX分院"]]]
df.loc[df["水表名"]==[["XXX花圃+"],["XXX成教院XXX分院"]]]
df.loc[(df["水表名"]==["XXX花圃+"]) |(df["水表名"]==["XXX成教院XXX分院"])]
df.水表名.isin(['JJ', 'NN'])
df.水表名.isin(['花圃', 'XXX成教院XXX分院'])
df.水表名.isin(['花圃', 'XXX成教院XXX分院']).word
df.水表名.isin(['花圃', 'XXX成教院XXX分院']).word()
df[df.水表名.isin(['花圃', 'XXX成教院XXX分院'])].word()
df[df.水表名.isin(['花圃', 'XXX成教院XXX分院'])]
cc=df[df.水表名.isin(['花圃', 'XXX成教院XXX分院'])]
cc=df[df.水表名.isin(['XXX花圃+', 'XXX成教院XXX分院'])]
cc=df[df.水表名.isin(['XXX花圃+', 'XXX成教院XXX分院',"XXX田径场厕所","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])]
cc.corr()
ccc=cc.corr()
cc=pd.read_excel("工作簿1.xlsx")
cc.corr()
ccc=cc.corr()
cc=pd.read_excel("工作簿1.xlsx")
ccc=cc.corr()
df.to_csv("原始.csv")
df
len(df["时间"].loc[(df["时间"]>2)&(df["时间"]<5)])
cc5=df["时间"].loc[(df["时间"]>2)&(df["时间"]<5)]
cc5=df.loc[(df["时间"]>2)&(df["时间"]<5)]
cc5.to_csv("两点-5.csv")
cc=cc5[cc5.水表名.isin(['XXX花圃+', 'XXX成教院XXX分院',"XXX田径场厕所","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])]
cc.to_csv("405.csv")
cc
cc["采集时间"]=pd.to_datetime(cc["采集时间"])
cc=pd.read_csv("405.csv")
cc=pd.read_csv("405.csv",encoding="gbk")
cc
cc=pd.read_csv("405.csv")
cc=pd.read_csv("405.csv",encoding="gbk")
cc
cc=pd.read_csv("405.csv",encoding="gbk")
cc
cc=pd.read_csv("405.csv",encoding="gbk")
cc
cc=pd.read_csv("405.csv",encoding="gbk")
cc
cc=cc5[cc5.水表名.isin(["校医南+","XXX校医院","车队+"])]
cc.to_csv("416.csv")
cc=cc5[cc5.水表号.isin(["3320100600","183671860","3320100700","3200200100","3390100400","3180700100"])]
cc.to_csv("404.csv")
cc=pd.read_csv("405.csv",encoding="gbk")
cc
cc=cc5[cc5.水表号.isin(["3620300200","0","3315400100","3620302700","3620302600","3000000001","3290100300","3422000100","3620302500","3280100100","3161100100","3160100100","3000000000","3421300300","3421900100","3421300200","3170200100","3170100600","3620302000","3620301200","3620301000","3620300900","3620300800","3620300700","3620300600","3620301100","3400500100","3421300100","3422200100","3315400200","3160300300","3620301300"])]
cc
cc.to_csv("401.csv")
cc=cc5[cc5.水表号.isin(["3620301800","3480300100","3480200100","3320100900","3450200100","3320100800"])]
cc.to_csv("402.csv")
cc=cc5[cc5.水表号.isin(["3620300300","3160200300","3620301400","3320100200","3480100100","3620301700","3320100100","3620302100","3620300400","3480400100","3320100500","1836718629","3320100400","3620301500","1836718625","3320100300","1836718633","3620302400","3620302300","3620301900","3390100500","3520300100","3360300100","3390100200","3390100300","3421400100"])]
cc.to_csv("403.csv")
cc=pd.read_csv("405.csv",encoding="gbk")
cc

ccc=cc.groupby(["水表名","采集时间"])["用量"].sum()
ccc.to_csv("405 15fenzhong.csv")
ccc=cc.groupby(["水表名","采集时间2"])["用量"].sum()
ccc.to_csv("405 15fenzhong.csv")
ccc=cc.groupby(["采集时间2"])["用量"].sum()
cc=cc5[cc5.水表名.isin(["XXX田径场厕所"])]
t401=pd.read_csv("t401.csv")
t402=pd.read_csv("t402.csv")
t403=pd.read_csv("t403.csv")
t404=pd.read_csv("t404.csv")
t405=pd.read_csv("t405.csv")
t416=pd.read_csv("t416.csv")
t401=pd.read_csv("t401.csv",encoding="gbk")
t402=pd.read_csv("t402.csv",encoding="gbk")
t403=pd.read_csv("t403.csv",encoding="gbk")
t404=pd.read_csv("t404.csv",encoding="gbk")
t405=pd.read_csv("t405.csv",encoding="gbk")
t416=pd.read_csv("t416.csv",encoding="gbk")
t401=pd.read_csv("t401.csv",encoding="gbk")
t401=pd.read_csv("t401.csv",encoding="gbk")
t402=pd.read_csv("t402.csv",encoding="gbk")
t403=pd.read_csv("t403.csv",encoding="gbk")
t404=pd.read_csv("t404.csv",encoding="gbk")
t405=pd.read_csv("t405.csv",encoding="gbk")
t416=pd.read_csv("t416.csv",encoding="gbk")
cc=cc5[t405.水表名.isin(["XXX田径场厕所"])]
cc=cc5[t405.水表号.isin(["3421200300"])]
cc=cc5[cc5.水表名.isin(["XXX田径场厕所"])]
cc=t401[t401.水表名.isin(["XXX田径场厕所"])]
cc=t405[t405.水表名.isin(["XXX田径场厕所"])]
cc=t405[t405.水表名.isin(["XXX田径场厕所"])]
ccc=t405[t405.水表名.isin(["XXX成教院XXX分院"])]
cccc=ccc["用量"]-cc["用量"]
print(ccc["用量"])
print(cc["用量"])
t401=pd.read_csv("t401.csv",encoding="gbk")
t402=pd.read_csv("t402.csv",encoding="gbk")
t403=pd.read_csv("t403.csv",encoding="gbk")
t404=pd.read_csv("t404.csv",encoding="gbk")
t405=pd.read_csv("t405.csv",encoding="gbk")
t416=pd.read_csv("t416.csv",encoding="gbk")
cc=t405[t405.水表名.isin(["XXX田径场厕所"])]
ccc=t405[t405.水表名.isin(["XXX成教院XXX分院"])]
cccc=ccc["用量"]-cc["用量"]
cc=t405[t405.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc=t405[t405.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
ccc=t405[t405.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
cccc=ccc["用量"]-cc["用量"]
cc=t405[t405.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
ccc=t405[t405.水表名.isin(["XXX花圃+"])].reset_index(drop=True)
cc=cc.groupby("采集时间2")["用量"].sum()
cc=cc.groupby("采集时间2")["用量"].sum().reset_index(drop=True)
cc=t405[t405.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc=cc.groupby("采集时间2")["用量"].sum().reset_index(drop=True)
cccc=ccc["用量"]-cc["用量"]
ccc["用量"]
cc["用量"]
cc
cccc=ccc["用量"]-cc
cc=t401[t401.水表名.isin(["纳米楼4.5楼+","纳米楼3楼+"])].reset_index(drop=True)
ccc=t401[t401.水表名.isin(["XXX国际纳米研究所"])].reset_index(drop=True)
cc=cc.groupby("采集时间2")["用量"].sum().reset_index(drop=True)
cccc=ccc["用量"]-cc
print(cccc)
ccc=t401[t401.水表名.isin(["XXX国际纳米研究所"])].reset_index(drop=True)
t401.水表名.isin(["XXX国际纳米研究所"])
ccc=t401[t401.水表名.isin(["XXX国际纳米研究所"])].reset_index(drop=True)
ccc=t401[t401.水表号.isin(["3315400100"])].reset_index(drop=True)
cc=t401[t401.水表名.isin(["纳米楼4.5楼+","纳米楼3楼+"])].reset_index(drop=True)
ccc=t401[t401.水表号.isin(["3315400100"])].reset_index(drop=True)
cc=cc.groupby("采集时间2")["用量"].sum().reset_index(drop=True)
cccc=ccc["用量"]-cc
print(cccc)
cc=t405[t405.水表名.isin(["XXX田径场厕所"])].set_index(["采集时间2"], inplace=True)
ccc=t405[t405.水表名.isin(["XXX成教院XXX分院"])].set_index(["采集时间2"], inplace=True)
cccc=ccc["用量"]-cc["用量"]
cc=t405[t405.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
ccc=t405[t405.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
cc=t405[t405.水表名.isin(["XXX田径场厕所"])].set_index(["采集时间2"], inplace=True)
cc.set_index(["采集时间2"], inplace=True)
cc=t405[t405.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间2"], inplace=True)
ccc.set_index(["采集时间2"], inplace=True)
cccc=ccc["用量"]-cc["用量"]
cc=t405[t405.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc.set_index(["采集时间2"], inplace=True)
ccc=t405[t405.水表名.isin(["XXX花圃+"])].reset_index(drop=True)
cc=cc.groupby("采集时间2")["用量"].sum().reset_index(drop=True)
ccc.set_index(["采集时间2"], inplace=True)
cccc=ccc["用量"]-cc
cc=cc.groupby("采集时间2")["用量"].sum().reset_index(drop=True)
cc=t405[t405.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc=cc.groupby("采集时间2")["用量"].sum().reset_index(drop=True)
ccc=t405[t405.水表名.isin(["XXX花圃+"])].reset_index(drop=True)
cc=t405[t405.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc.set_index(["采集时间2"], inplace=True)
cc=cc.groupby("采集时间2")["用量"].sum().reset_index(drop=True)
cc=cc.groupby("采集时间2")["用量"].sum()
cc=t405[t405.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc=cc.groupby("采集时间2")["用量"].sum()
cc=t405[t405.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc.set_index(["采集时间2"], inplace=True)
ccc=t405[t405.水表名.isin(["XXX花圃+"])]
cc=cc.groupby("采集时间2")["用量"].sum()
ccc.set_index(["采集时间2"], inplace=True)
cccc=ccc["用量"]-cc
cc=t401[t401.水表名.isin(["纳米楼4.5楼+","纳米楼3楼+"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表号.isin(["3315400100"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["用量"].sum()

cccc=ccc["用量"]-cc
print(cccc)
cc=t401[t401.水表名.isin(["XXXT馆后平房","XXX后勤楼","校管中心种子楼东+","XXX图书馆","XXX毒物研究所","XXX种子楼"])]
cc.set_index(["采集时间2"], inplace=True)
ccc=t401[t401.水表名.isin(["区域2"])]
ccc.set_index(["采集时间2"], inplace=True)
cc=cc.groupby("采集时间2")["用量"].sum()
cccc=ccc["用量"]-cc
cc=t401[t401.水表名.isin(["XXX大楼厕所西","XXX科学楼","XXX大楼厕所东","XXX中心水池","XXX西大楼"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表名.isin(["区域1（西）"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["用量"].sum()

cccc=ccc["用量"]-cc
cc=t401[t401.水表名.isin(["高配房+"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表名.isin(["XXX植物园"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["用量"].sum()

cccc=ccc["用量"]-cc
t401["cha"]=t401["当前读数"]-t401["上次读数"]
t402["cha"]=t401["当前读数"]-t401["上次读数"]
t403["cha"]=t401["当前读数"]-t401["上次读数"]
t404["cha"]=t401["当前读数"]-t401["上次读数"]
t405["cha"]=t401["当前读数"]-t401["上次读数"]
t416["cha"]=t401["当前读数"]-t401["上次读数"]
cc=t401[t401.水表名.isin(["司法鉴定中心","XXX国际纳米研究所","区域2","区域1（西）","书店+","新大门传达室+","养殖馆附房保卫处宿舍+","养殖馆公共厕所+","养殖馆附房二楼厕所+","养殖馆附房一楼厕所+","养殖馆+","养殖队+","XXX教学大楼总表","XXX中心大楼泵房","XXX东大楼","XXXM馆","XXX植物园"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表名.isin(["XXX植物园"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["用量"].sum()

cccc=ccc["用量"]-cc
cc=t401[t401.水表名.isin(["司法鉴定中心","XXX国际纳米研究所","区域2","区域1（西）","书店+","新大门传达室+","养殖馆附房保卫处宿舍+","养殖馆公共厕所+","养殖馆附房二楼厕所+","养殖馆附房一楼厕所+","养殖馆+","养殖队+","XXX教学大楼总表","XXX中心大楼泵房","XXX东大楼","XXXM馆","XXX植物园"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表名.isin(["XXX植物园"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t401[t401.水表名.isin(["司法鉴定中心","XXX国际纳米研究所","区域2","区域1（西）","书店+","新大门传达室+","养殖馆附房保卫处宿舍+","养殖馆公共厕所+","养殖馆附房二楼厕所+","养殖馆附房一楼厕所+","养殖馆+","养殖队+","XXX教学大楼总表","XXX中心大楼泵房","XXX东大楼","XXXM馆","XXX植物园"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表名.isin(["养殖队6721副表+"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
t401["cha"]=t401["当前读数"]-t401["上次读数"]
t402["cha"]=t402["当前读数"]-t402["上次读数"]
t403["cha"]=t403["当前读数"]-t403["上次读数"]
t404["cha"]=t404["当前读数"]-t404["上次读数"]
t405["cha"]=t405["当前读数"]-t405["上次读数"]
t416["cha"]=t416["当前读数"]-t416["上次读数"]
cc=t401[t401.水表名.isin(["司法鉴定中心","XXX国际纳米研究所","区域2","区域1（西）","书店+","新大门传达室+","养殖馆附房保卫处宿舍+","养殖馆公共厕所+","养殖馆附房二楼厕所+","养殖馆附房一楼厕所+","养殖馆+","养殖队+","XXX教学大楼总表","XXX中心大楼泵房","XXX东大楼","XXXM馆","XXX植物园"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表名.isin(["养殖队6721副表+"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t401[t401.水表名.isin(["XXX8舍热泵"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表名.isin(["XXX第八学生宿舍"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t401[t401.水表号.isin(["183671860"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表名.isin(["XXX第八学生宿舍"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t404[t404.水表号.isin(["183671860"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t404[t404.水表名.isin(["XXX第八学生宿舍"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t416[t416.水表号.isin(["3620303100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t416[t416.水表名.isin(["XXX校医院"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t416[t416.水表号.isin(["3290100100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t416[t416.水表名.isin(["校医院南+"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t416[t416.水表号.isin(["3290100100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t416[t416.水表号.isin(["3620300500"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=cc5[cc5.水表名.isin(["校医南+","XXX校医院","车队+"])]
cc=cc5[cc5.水表号.isin(["3620300500","3290100100","3620303100"])]
cc.to_csv("416.csv")
t416=pd.read_csv("t416.csv",encoding="gbk")
t416["cha"]=t416["当前读数"]-t416["上次读数"]
cc=t416[t416.水表号.isin(["3290100100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t416[t416.水表号.isin(["3290100100"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t416[t416.水表号.isin(["3620303100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t416[t416.水表名.isin(["XXX校医院"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
 #2jin 1
cc=t416[t416.水表号.isin(["3290100100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t416[t416.水表号.isin(["3290100100"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t416[t416.水表号.isin(["3290100100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t416[t416.水表号.isin(["3620300500"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t416[t416.水表号.isin(["1836718629"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t416[t416.水表号.isin(["3320100500"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t403[t403.水表号.isin(["1836718629"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t403[t403.水表号.isin(["3320100500"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t403[t403.水表号.isin(["3620301500","1836718625"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t403[t403.水表号.isin(["3320100400"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t403[t403.水表号.isin(["1836718633"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t403[t403.水表号.isin(["3320100300"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t403[t403.水表号.isin(["3620301500","1836718625"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t403[t403.水表号.isin(["3320100400"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t403[t403.水表号.isin(["3320100200","3480100100","3620301700","3320100100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t403[t403.水表号.isin(["3620301400"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t403[t403.水表号.isin(["3480400100","3320100500","3320100400","3320100300"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t403[t403.水表号.isin(["3620300400"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t403[t403.水表号.isin(["3160200300","3620301400","3620302100","3620300400","3620302400","3620302300","3620301900","3390100500","3520300100","3360300100","3390100200","3390100300","3421400100"])]
cc.set_index(["采集时间2"], inplace=True)
ccc=t403[t403.水表号.isin(["3620300400"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cc=t405[t405.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间2"], inplace=True)
ccc=t405[t405.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
ccc.set_index(["采集时间2"], inplace=True)
cccc=ccc["用量"]-cc["用量"]
cccc.to_csv("cccc.csv")
cc=t405[t405.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc.set_index(["采集时间2"], inplace=True)
ccc=t405[t405.水表名.isin(["XXX花圃+"])]
cc=cc.groupby("采集时间2")["用量"].sum()
ccc.set_index(["采集时间2"], inplace=True)
cccc=ccc["用量"]-cc
cccc.to_csv("cccc.csv")
cc=t401[t401.水表名.isin(["XXXT馆后平房","XXX后勤楼","校管中心种子楼东+","XXX图书馆","XXX毒物研究所","XXX种子楼"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表名.isin(["区域2"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["用量"].sum()

cccc=ccc["用量"]-cc

cccc.to_csv("cccc.csv")
cc=t401[t401.水表名.isin(["纳米楼4.5楼+","纳米楼3楼+"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表号.isin(["3315400100"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["用量"].sum()

cccc=ccc["用量"]-cc
cccc.to_csv("cccc.csv")
cc=t401[t401.水表名.isin(["XXX大楼厕所西","XXX科学楼","XXX大楼厕所东","XXX中心水池","XXX西大楼"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表名.isin(["区域1（西）"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["用量"].sum()

cccc=ccc["用量"]-cc
cccc.to_csv("cccc.csv")
cc=t401[t401.水表名.isin(["高配房+"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表名.isin(["XXX植物园"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["用量"].sum()

cccc=ccc["用量"]-cc

cccc.to_csv("cccc.csv")
cc=t401[t401.水表名.isin(["司法鉴定中心","XXX国际纳米研究所","区域2","区域1（西）","书店+","新大门传达室+","养殖馆附房保卫处宿舍+","养殖馆公共厕所+","养殖馆附房二楼厕所+","养殖馆附房一楼厕所+","养殖馆+","养殖队+","XXX教学大楼总表","XXX中心大楼泵房","XXX东大楼","XXXM馆","XXX植物园"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t401[t401.水表名.isin(["养殖队6721副表+"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cccc.to_csv("cccc.csv")
cc=t404[t404.水表号.isin(["183671860"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t404[t404.水表名.isin(["XXX第八学生宿舍"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cccc.to_csv("cccc.csv")
cc=t416[t416.水表号.isin(["3620303100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t416[t416.水表名.isin(["XXX校医院"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cccc.to_csv("cccc.csv")
cc=t416[t416.水表号.isin(["3290100100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t416[t416.水表号.isin(["3620300500"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cccc.to_csv("cccc.csv")
cc=t403[t403.水表号.isin(["1836718629"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t403[t403.水表号.isin(["3320100500"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cccc.to_csv("cccc.csv")
cc=t403[t403.水表号.isin(["3620301500","1836718625"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t403[t403.水表号.isin(["3320100400"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cccc.to_csv("cccc.csv")
cc=t403[t403.水表号.isin(["1836718633"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t403[t403.水表号.isin(["3320100300"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cccc.to_csv("cccc.csv")
cc=t403[t403.水表号.isin(["3320100200","3480100100","3620301700","3320100100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t403[t403.水表号.isin(["3620301400"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cccc.to_csv("cccc.csv")
cc=t403[t403.水表号.isin(["3480400100","3320100500","3320100400","3320100300"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t403[t403.水表号.isin(["3620300400"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cccc.to_csv("cccc.csv")
cc=t403[t403.水表号.isin(["3160200300","3620301400","3620302100","3620300400","3620302400","3620302300","3620301900","3390100500","3520300100","3360300100","3390100200","3390100300","3421400100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t403[t403.水表号.isin(["3620300300"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cccc.to_csv("cccc.csv")
t416=pd.read_csv("t416.csv",encoding="gbk")
cc5=pd.read_excel("无流量重复（总结）.xlsx")
cc6=cc5.groupby(["水表","时间"])["cha"].value_counts()
cc6
cc5=pd.read_excel("无流量重复（总结）.xlsx")
cc6=cc5.groupby(["水表"])["cha"].value_counts()
cc6
cc6.to_csv("chuchong .csv")
cc5=pd.read_excel("无流量重复（总结）.xlsx")
cc6=cc5.groupby(["水表"])["cha"].value_counts()
cc6
cc5=pd.read_excel("无流量重复（总结）.xlsx")
cc6=cc5.groupby(["水表","时间"])["cha"].value_counts()
cc6
cc5=pd.read_excel("无流量重复（总结）.xlsx")
cc6=cc5.groupby(["水表","时间"])["cha"].value_counts()
cc6
cc5=pd.read_excel("无流量重复（总结）.xlsx")
cc6=cc5.groupby(["水表","时间"])["cha"].value_counts()
cc6
cc6.to_csv("chuchong .csv")
cc6=cc5.groupby(["水表","时间"])["cha"].sum()
cc5=pd.read_excel("无流量重复（总结）.xlsx")
cc5=pd.read_excel("无流量重复（总结）.xlsx")
cc6=cc5.groupby(["水表","时间"])["cha"].value_counts()
cc6.to_csv("chuchong .csv")
cc5=pd.read_excel("无流量重复（总结）.xlsx")
cc6=cc5.groupby(["水表","时间"])["cha"].value_counts()
cc6.to_csv("chuchong .csv")
cc=t401.groupby(["水表名","采集时间"])["cha"].max()
cc1=t401.groupby(["水表名","采集时间"])["cha"].mean()
cc=t401.groupby(["水表名","采集时间"])["cha"].max()
ccq=t401.groupby(["水表名","采集时间"])["cha"].mean()
ccw=ccq/cc
cce=t401.groupby(["水表名","采集时间"])["cha"].count()
cce=t401.groupby(["水表名","采集时间"])["cha"].value_counts()
cce.to_csv("cce.csv")
import pandas as pd 

import numpy as np
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
cc1=pd.read_excel("一季度.xlsx")
cc2=pd.read_excel("二季度.xlsx")
cc3=pd.read_excel("三季度.xlsx")
cc4=pd.read_excel("四季度.xlsx")
cc1=pd.read_csv("附件_一季度.csv",encoding="gbk")
cc2=pd.read_csv("附件_二季度.csv",encoding="gbk")
cc3=pd.read_csv("附件_三季度.csv",encoding="gbk")
cc4=pd.read_csv("附件_四季度.csv",encoding="gbk")

df = pd.concat([cc1, cc2,cc3,cc4])
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
df["cha"]=(df["当前读数"]-df["上次读数"]-df["用量"])
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间2"], inplace=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
ccc.set_index(["采集时间2"], inplace=True)
ccc.set_index(["采集时间"], inplace=True)
cccc=ccc["cha"]-cc["cha"]
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
ccc.set_index(["采集时间"], inplace=True)
cccc=ccc["cha"]-cc["cha"]
cccc.to_csv("cccc.csv")
ccc["cha1"]=ccc["cha"]-cc["cha"]
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX花圃+"])]
cc=cc.groupby("采集时间")["用量"].sum()
ccc.set_index(["采集时间"], inplace=True)
ccc["cha1"]=ccc["cha"]-cc["cha"]
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX花圃+"])]
cc=cc.groupby("采集时间")["用量"].sum()
ccc.set_index(["采集时间"], inplace=True)
ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX花圃+"])]
cc=cc.groupby("采集时间")["cha"].sum()
ccc.set_index(["采集时间"], inplace=True)
ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXXT馆后平房","XXX后勤楼","校管中心种子楼东+","XXX图书馆","XXX毒物研究所","XXX种子楼"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["区域2"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXXT馆后平房","XXX后勤楼","校管中心种子楼东+","XXX图书馆","XXX毒物研究所","XXX种子楼"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["区域2"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["纳米楼4.5楼+","纳米楼3楼+"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3315400100"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["用量"].sum()

ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX大楼厕所西","XXX科学楼","XXX大楼厕所东","XXX中心水池","XXX西大楼"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["区域1（西）"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["用量"].sum()

ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["高配房+"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["XXX植物园"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["司法鉴定中心","XXX国际纳米研究所","区域2","区域1（西）","书店+","新大门传达室+","养殖馆附房保卫处宿舍+","养殖馆公共厕所+","养殖馆附房二楼厕所+","养殖馆附房一楼厕所+","养殖馆+","养殖队+","XXX教学大楼总表","XXX中心大楼泵房","XXX东大楼","XXXM馆","XXX植物园"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["养殖队6721副表+"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()


ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["司法鉴定中心","XXX国际纳米研究所","区域2","区域1（西）","书店+","新大门传达室+","养殖馆附房保卫处宿舍+","养殖馆公共厕所+","养殖馆附房二楼厕所+","养殖馆附房一楼厕所+","养殖馆+","养殖队+","XXX教学大楼总表","XXX中心大楼泵房","XXX东大楼","XXXM馆","XXX植物园"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["养殖队6721副表+"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()


ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["183671860"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["XXX第八学生宿舍"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()


ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["3620303100"])]
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX校医院"])]
ccc.set_index(["采集时间"], inplace=True)
cc=cc.groupby("采集时间")["cha"].sum()
cccc=ccc["cha"]-cc
cccc.to_csv("cccc.csv")
df["cha"]=(df["当前读数"]-df["上次读数"]
df["cha"]=(df["当前读数"]-df["上次读数"])
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
ccc.set_index(["采集时间"], inplace=True)
ccc["cha1"]=ccc["cha"]-cc["cha"]
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX花圃+"])]
cc=cc.groupby("采集时间")["cha"].sum()
ccc.set_index(["采集时间"], inplace=True)
ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXXT馆后平房","XXX后勤楼","校管中心种子楼东+","XXX图书馆","XXX毒物研究所","XXX种子楼"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["区域2"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXXT馆后平房","XXX后勤楼","校管中心种子楼东+","XXX图书馆","XXX毒物研究所","XXX种子楼"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["区域2"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()
ccc["cc"]=cc
ccc["cha1"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
ccc.set_index(["采集时间"], inplace=True)
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc["cha"]
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
cc=cc.groupby("采集时间")["cha"].sum()

ccc.set_index(["采集时间"], inplace=True)
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc["cha"]
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc=cc.groupby("采集时间")["cha"].sum()
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
cc=cc.groupby("采集时间")["cha"].sum()

ccc.set_index(["采集时间"], inplace=True)
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc["cha"]
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
cc=cc.groupby("采集时间")["cha"].sum()

ccc.set_index(["采集时间"], inplace=True)
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
cc=cc.groupby("采集时间")["cha"].sum()

ccc.set_index(["采集时间"], inplace=True)
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX花圃+"])]
cc=cc.groupby("采集时间")["cha"].sum()
ccc.set_index(["采集时间"], inplace=True)
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXXT馆后平房","XXX后勤楼","校管中心种子楼东+","XXX图书馆","XXX毒物研究所","XXX种子楼"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["区域2"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
cc=cc.groupby("采集时间")["cha"].sum()

ccc.set_index(["采集时间"], inplace=True)
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX花圃+"])]
cc=cc.groupby("采集时间")["cha"].sum()
ccc.set_index(["采集时间"], inplace=True)
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXXT馆后平房","XXX后勤楼","校管中心种子楼东+","XXX图书馆","XXX毒物研究所","XXX种子楼"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["区域2"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["纳米楼4.5楼+","纳米楼3楼+"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3315400100"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc["cha"]
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["纳米楼4.5楼+","纳米楼3楼+"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3315400100"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX大楼厕所西","XXX科学楼","XXX大楼厕所东","XXX中心水池","XXX西大楼"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["区域1（西）"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["高配房+"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["XXX植物园"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
cc=cc.groupby("采集时间")["cha"].sum()

ccc.set_index(["采集时间"], inplace=True)
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["司法鉴定中心","XXX国际纳米研究所","区域2","区域1（西）","书店+","新大门传达室+","养殖馆附房保卫处宿舍+","养殖馆公共厕所+","养殖馆附房二楼厕所+","养殖馆附房一楼厕所+","养殖馆+","养殖队+","XXX教学大楼总表","XXX中心大楼泵房","XXX东大楼","XXXM馆","XXX植物园"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["养殖队6721副表+"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX田径场厕所"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX成教院XXX分院"])].reset_index(drop=True)
cc=cc.groupby("采集时间")["cha"].sum()

ccc.set_index(["采集时间"], inplace=True)
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX成教院XXX分院","离退休活动室","XXXL馆","XXXL楼","XXXS馆","XXXK","XXXK酒店","XXX体育馆","XXX干训楼"])].reset_index(drop=True)
cc.set_index(["采集时间"], inplace=True)
ccc=df[df.水表名.isin(["XXX花圃+"])]
cc=cc.groupby("采集时间")["cha"].sum()
ccc.set_index(["采集时间"], inplace=True)
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXXT馆后平房","XXX后勤楼","校管中心种子楼东+","XXX图书馆","XXX毒物研究所","XXX种子楼"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["区域2"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["纳米楼4.5楼+","纳米楼3楼+"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3315400100"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["XXX大楼厕所西","XXX科学楼","XXX大楼厕所东","XXX中心水池","XXX西大楼"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["区域1（西）"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["高配房+"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["XXX植物园"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表名.isin(["司法鉴定中心","XXX国际纳米研究所","区域2","区域1（西）","书店+","新大门传达室+","养殖馆附房保卫处宿舍+","养殖馆公共厕所+","养殖馆附房二楼厕所+","养殖馆附房一楼厕所+","养殖馆+","养殖队+","XXX教学大楼总表","XXX中心大楼泵房","XXX东大楼","XXXM馆","XXX植物园"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["养殖队6721副表+"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["183671860"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["XXX第八学生宿舍"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["3620303100"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表名.isin(["XXX校医院"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=t416[t416.水表号.isin(["3290100100"])]
cc.set_index(["采集时间2"], inplace=True)

ccc=t416[t416.水表号.isin(["3620300500"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["3290100100"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3620300500"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["3290100100"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3620300500"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["1836718629"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3320100500"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc["cha"]
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["1836718629"])]
cc.set_index(["采集时间"], inplace=True)
cc=cc.groupby("采集时间")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["3620301500","1836718625"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3320100400"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc["cha"]
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["3620301500","1836718625"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3320100400"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["1836718633"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3320100300"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["3320100200","3480100100","3620301700","3320100100"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3620301400"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["3480400100","3320100500","3320100400","3320100300"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3620300400"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()

ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["3160200300","3620301400","3620302100","3620300400","3620302400","3620302300","3620301900","3390100500","3520300100","3360300100","3390100200","3390100300","3421400100"])]
cc.set_index(["采集时间"], inplace=True)

ccc=df[df.水表号.isin(["3620300300"])]
ccc.set_index(["采集时间"], inplace=True)

cc=cc.groupby("采集时间")["cha"].sum()
ccc["下级合"]=cc

ccc["上下级差"]=ccc["cha"]-cc
ccc.to_csv("cccc.csv")
cc=df[df.水表号.isin(["3160200300","3620301400","3620302100","3620300400","3620302400","3620302300","3620301900","3390100500","3520300100","3360300100","3390100200","3390100300","3421400100"])]
cc=df[df.水表号.isin(["3620303100","3421200300","3363000100","3313800500","3370100100","3313200200","3370300100","3370200100","3030100100","3210100100","183671860","3320100700","3200200100","3390100400","3180700100","3160200300","3320100200","3480100100","3620301700","3320100100","3620302100","3480400100","1836718629","3620301500","1836718625","1836718633","3620302400","3620302300","3620301900","3390100500","3520300100","3360300100","3390100200","3390100300","3421400100","3620301800","3480300100","3480200100","3320100900","3450200100","3320100800","0","3620302700","3620302600","3290100300","3422000100","3620302500","3280100100","3161100100","3160100100","3421300300","3421900100","3421300200","3170200100","3170100600","3620302000","3620301200"])]
cc=df[df.水表号.isin(["3620303100","3421200300","3363000100","3313800500","3370100100","3313200200","3370300100","3370200100","3030100100","3210100100","183671860","3320100700","3200200100","3390100400","3180700100","3160200300","3320100200","3480100100","3620301700","3320100100","3620302100","3480400100","1836718629","3620301500","1836718625","1836718633","3620302400","3620302300","3620301900","3390100500","3520300100","3360300100","3390100200","3390100300","3421400100","3620301800","3480300100","3480200100","3320100900","3450200100","3320100800","0","3620302700","3620302600","3290100300","3422000100","3620302500","3280100100","3161100100","3160100100","3421300300","3421900100","3421300200","3170200100","3170100600","3620302000","3620301200","3620301000","3620300900","3620300800","3620300700","3620300600","3620301100","3400500100","3421300100","3422200100","3315400200","3620301300","3620303000","3620302900","3620302800","3030100102","3030100101","3312800100","3620302200"])]
cc.to_csv("无下级.csv")
cc=t401.groupby(["水表名","采集时间"])["cha"].max()
t401=pd.read_csv("t401.csv",encoding="gbk")
t402=pd.read_csv("t402.csv",encoding="gbk")
t403=pd.read_csv("t403.csv",encoding="gbk")
t404=pd.read_csv("t404.csv",encoding="gbk")
t405=pd.read_csv("t405.csv",encoding="gbk")
t416=pd.read_csv("t416.csv",encoding="gbk")
cc=t401.groupby(["水表名","采集时间"])["cha"].max()
t401["cha"]=t401["当前读数"]-t401["上次读数"]
t402["cha"]=t402["当前读数"]-t402["上次读数"]
t403["cha"]=t403["当前读数"]-t403["上次读数"]
t404["cha"]=t404["当前读数"]-t404["上次读数"]
t405["cha"]=t405["当前读数"]-t405["上次读数"]
t416["cha"]=t416["当前读数"]-t416["上次读数"]
cc=t401.groupby(["水表名","采集时间"])["cha"].max()
ccq=t401.groupby(["水表名","采集时间"])["cha"].mean()
ccw=ccq/cc
cce=t401.groupby(["水表名","采集时间"])["cha"].max().value_counts()
cc=t401.groupby(["水表名","采集时间"])["cha"].max()
ccq=t401.groupby(["水表名","采集时间"])["cha"].mean()
ccw=ccq/cc
cce.to_csv("cce.csv")
ccq.to_csv("ccq.csv")
ccw.to_csv("ccw.csv")
cc=t401.groupby(["水表名","采集时间"])["cha"].max()
ccq=t401.groupby(["水表名","采集时间"])["cha"].mean()
ccw=ccq/cc
ccc.to_csv("cc.csv")
ccq.to_csv("ccq.csv")
ccw.to_csv("ccw.csv")
cc=t401.groupby(["水表名","采集时间"])["cha"].max()
cc.to_csv("cc.csv")
cc=df.groupby(["水表名","采集时间"])["cha"].max()
ccq=df.groupby(["水表名","采集时间"])["cha"].mean()
ccw=ccq/cc
cc.to_csv("cc.csv")
ccq.to_csv("ccq.csv")
ccw.to_csv("ccw.csv")
df["采集时间"]=time.strptime(df["采集时间"], "%Y-%m-%d")
import time
df["采集时间"]=time.strptime(df["采集时间"], "%Y-%m-%d")
df["采集时间"]=time.strptime(str(df["采集时间"]), "%Y-%m-%d")
import datetime
df["采集时间"]=datetime.strptime(df["采集时间"],'%Y/%M/%D')
df['采集时间']=df['采集时间'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
from datetime import datetime
df['采集时间']=df['采集时间'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
df['采集时间']=df['采集时间'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d hh:mm:ss').strftime('%Y/%m/%d'))
df['采集时间']=df['采集时间'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d hh:mm').strftime('%Y/%m/%d'))
cc1=pd.read_excel("一季度.xlsx")
cc2=pd.read_excel("二季度.xlsx")
cc3=pd.read_excel("三季度.xlsx")
cc4=pd.read_excel("四季度.xlsx")
reset
import pandas as pd 

import numpy as np
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
cc1=pd.read_csv("一季度.csv",encoding="gbk")
cc2=pd.read_csv("二季度.csv",encoding="gbk")
cc3=pd.read_csv("三季度.csv",encoding="gbk")
cc4=pd.read_csv("四季度.csv",encoding="gbk")
df = pd.concat([cc1, cc2,cc3,cc4])
df["cha"]=df["当前读数"]-df["上次读数"]
cc=df.groupby(["水表名","采集时间"])["cha"].max()
ccq=df.groupby(["水表名","采集时间"])["cha"].mean()
ccw=ccq/cc
cc.to_csv("cc.csv")
ccq.to_csv("ccq.csv")
ccw.to_csv("ccw.csv")
cc=df.groupby(["水表名","采集时间"])["cha"].value_counts().max()
cc=df.groupby(["水表名","采集时间"])["cha"].value_counts()
cc=df.groupby(["水表名","采集时间"])["cha"].max()
cc=df.groupby(["水表名","采集时间"])["cha"].value_counts()
cc.to_csv("cc.csv")

cc=df.groupby(["水表名","采集时间"])["cha"].value_counts()
cc.to_csv("cc.csv")

ccq=df.groupby(["水表名","采集时间"])["cha"].median()
cc=df.groupby(["水表名","采集时间"])["cha"].median()
cc.index
for i in cc.index:
    print(i)
    
cc.to_csv("cc.csv")
ccq=df.groupby(["采集时间"])["cha"]-cc
ccq=df.groupby(["采集时间"])["cha"]
len(ccq)
ccq=df.groupby(["水表名"])["cha"]
len(ccq)
ccq.to_csv("ccq.csv")
ccq=df.groupby(["水表名","采集时间"])["cha"].value_counts()
ccq
ccq.to_csv("ccq.csv")
cccc=df[df.水表号.isin(["3520300100","3360300100","3390100200","3390100300","3421400100","3620301800","3480300100","3480200100","3320100900","3450200100","3320100800","0","3620302700","3620302600","3290100300","3422000100","3620302500","3280100100","3161100100","3160100100","3421300300","3421900100","3421300200","3170200100","3170100600","3620302000","3620301200","3620301000","3620300900","3620300800","3620300700","3620300600","3620301100","3400500100","3421300100","3422200100","3315400200","3620301300","3620303000","3620302900","3620302800","3030100102","3030100101","3312800100","3620302200"])]
cc=df[df.水表号.isin(["3620303100","3421200300","3363000100","3313800500","3370100100","3313200200","3370300100","3370200100","3030100100","3210100100","183671860","3320100700","3200200100","3390100400","3180700100","3160200300","3320100200","3480100100","3620301700","3320100100","3620302100","3480400100","1836718629","3620301500","1836718625","1836718633","3620302400","3620302300","3620301900","3390100500"])]
cc=df[df.水表号.isin(["3620303100","3421200300","3363000100","3313800500","3370100100","3313200200","3370300100","3370200100","3030100100","3210100100","183671860","3320100700","3200200100","3390100400","3180700100","3160200300","3320100200","3480100100","3620301700","3320100100","3620302100","3480400100","1836718629"])]
cccc=df[df.水表号.isin(["3620301500","1836718625","1836718633","3620302400","3620302300","3620301900","3390100500","3520300100","3360300100","3390100200","3390100300","3421400100","3620301800","3480300100","3480200100","3320100900","3450200100","3320100800","0","3620302700","3620302600","3290100300"])]
ccccc=df[df.水表号.isin(["3422000100","3620302500","3280100100","3161100100","3160100100","3421300300","3421900100","3421300200","3170200100","3170100600","3620302000","3620301200","3620301000","3620300900","3620300800","3620300700","3620300600","3620301100","3400500100","3421300100","3422200100","3315400200","3620301300","3620303000","3620302900","3620302800","3030100102","3030100101","3312800100","3620302200"])]
cc=df[df.水表号.isin(["3620303100","3421200300","3363000100","3313800500","3370100100","3313200200","3370300100","3370200100","3030100100","3210100100","183671860","3320100700","3200200100","3390100400","3180700100","3160200300","3320100200","3480100100","3620301700","3320100100","3620302100","3480400100","1836718629"])]
cccc=df[df.水表号.isin(["3620301500","1836718625","1836718633","3620302400","3620302300","3620301900","3390100500","3520300100","3360300100","3390100200","3390100300","3421400100","3620301800","3480300100","3480200100","3320100900","3450200100","3320100800","0","3620302700","3620302600","3290100300"])]
ccccc=df[df.水表号.isin(["3422000100","3620302500","3280100100","3161100100","3160100100","3421300300","3421900100","3421300200","3170200100","3170100600","3620302000","3620301200","3620301000","3620300900","3620300800","3620300700","3620300600","3620301100","3400500100","3421300100","3422200100"])]
ccccc=df[df.水表号.isin(["3315400200","3620301300","3620303000","3620302900","3620302800","3030100102","3030100101","3312800100","3620302200"])]
cc=df[df.水表号.isin(["3620303100","3421200300","3363000100","3313800500","3370100100","3313200200","3370300100","3370200100","3030100100","3210100100","183671860","3320100700","3200200100","3390100400","3180700100","3160200300","3320100200","3480100100","3620301700","3320100100","3620302100","3480400100","1836718629"])]
cccc=df[df.水表号.isin(["3620301500","1836718625","1836718633","3620302400","3620302300","3620301900","3390100500","3520300100","3360300100","3390100200","3390100300","3421400100","3620301800","3480300100","3480200100","3320100900","3450200100","3320100800","0","3620302700","3620302600","3290100300"])]
ccccc=df[df.水表号.isin(["3422000100","3620302500","3280100100","3161100100","3160100100","3421300300","3421900100","3421300200","3170200100","3170100600","3620302000","3620301200","3620301000","3620300900","3620300800","3620300700","3620300600","3620301100","3400500100","3421300100","3422200100"])]
cccccc=df[df.水表号.isin(["3315400200","3620301300","3620303000","3620302900","3620302800","3030100102","3030100101","3312800100","3620302200"])]


cc.to_csv("无下级.csv")
cccc.to_csv("无下级1.csv")
ccccc.to_csv("无下级2.csv")
cccccc.to_csv("无下级3.csv")
bb=df.loc[(df["采集时间"]>2019/1/1) ]
bb=df.loc[(df["采集时间"]>time.strptime(2019/1/1 00:00:00, "%Y/%m/%d  %H:%M:%S")) ]
bb=df.loc[(df["采集时间"]>time.strptime(2019/1/100:00:00, "%Y/%m/%d  %H:%M:%S")) ]
bb=df.loc[(df["采集时间"]> '2002-1-1 01:00:00')]
df
cc11=pd.read_csv("lv9.13.csv",encoding="gbk")
cc12=pd.concat([cc,ccc,ccccc,cccccc])
cc12=pd.concat([cc,cccc,ccccc,cccccc])
cc13=pd.concat([cc11,cc12])
cc14=cc13.loc[(cc13["时间"]>2) & (cc13["时间"]<5)]
cc14.to_csv("cc14 yejian .csv")
cc11=pd.read_csv("lv9.14.csv",encoding="gbk")
cc13=pd.concat([cc11,cc12])
cc14=cc13.loc[(cc13["时间"]>2) & (cc13["时间"]<5)]
cc14.set_index(["采集时间"])
cc14=cc14.set_index(["采集时间"])
cc14=cc13.loc[(cc13["时间"]>2) & (cc13["时间"]<5)]
cc14.groupby(["水表名","采集时间"])["cha"].value_count()
cc14.groupby(["水表名","采集时间"])["cha"].value_counts()
cc15=cc14.groupby(["水表名","采集时间"])["cha"].value_counts()
cc15.to_excel("cc15.xlsx")
cc=df.groupby(["水表名","采集时间"])["cha"].median()
ccq=df.groupby(["水表名","采集时间"])["cha"].value_counts()
cc=pd.read_csv("cc.csv")
ccq=pd.read_csv("ccq.csv")
cc=pd.read_csv("cc.csv",encoding="gbk")
ccq=pd.read_csv("ccq.csv",encoding="gbk")
cce=pd.merge(cc,ccq,on="时间")
print(cce)
cce=pd.merge(cc,ccq,on=["时间","表名"])
cc=pd.read_csv("cc.csv",encoding="gbk")
ccq=pd.read_csv("ccq.csv",encoding="gbk")
cce=pd.merge(cc,ccq,on=["时间","表名"])
cc=pd.read_csv("cc.csv",encoding="gbk")
ccq=pd.read_csv("ccq.csv",encoding="gbk")
cce=pd.merge(cc,ccq,on=["时间","表"])
cc11=pd.read_csv("lv9.14.csv",encoding="gbk")
cc13=pd.concat([cc11,cc12])
cc14=cc13.loc[(cc13["时间"]>2) & (cc13["时间"]<5)]
cc15=cc14.groupby(["水表名","采集时间"])["cha"].value_counts()
cc15.to_excel("cc15.xlsx")
cce.to_csv("cce.csv")
cce=pd.read_csv("cce.csv")
cc1=pd.read_csv("一季度.csv",encoding="gbk")
cc2=pd.read_csv("二季度.csv",encoding="gbk")
cc3=pd.read_csv("三季度.csv",encoding="gbk")
cc4=pd.read_csv("四季度.csv",encoding="gbk")
df = pd.concat([cc1, cc2,cc3,cc4])
df["cha"]=df["当前读数"]-df["上次读数"]
cc=df.groupby(["水表名","采集时间"])["cha"].mean()
ccq=df.groupby(["水表名","采集时间"])["cha"].value_counts()
cc1=pd.read_csv("一季度.csv",encoding="gbk")
cc2=pd.read_csv("二季度.csv",encoding="gbk")
cc3=pd.read_csv("三季度.csv",encoding="gbk")
cc4=pd.read_csv("四季度.csv",encoding="gbk")
df = pd.concat([cc1, cc2,cc3,cc4])
df["cha"]=df["当前读数"]-df["上次读数"]

cc=df.groupby(["水表名","采集时间"])["cha"].mean()
ccq=df.groupby(["水表名","采集时间"])["cha"].value_counts()
cc.to_csv("cc.csv")
ccq.to_csv("ccq.csv")
cc=pd.read_csv("cc.csv",encoding="gbk")
ccq=pd.read_csv("ccq.csv",encoding="gbk")
cce=pd.merge(cc,ccq,on=["时间","表"])
cce.to_csv("cce.csv")
cce=pd.read_csv("cce.csv",encoding="gbk")
ccr=cce.groupby(["水表名","时间"])["结果"].sum()
ccr=cce.groupby(["表","时间"])["结果"].sum()
ccr.to_csv("ccr.csv")
ccr=pd.read_csv("ccr.csv")
ccr=pd.read_csv("ccr.csv",encoding="gbk")
ming=ccr["表"].unique()
print(ming)
cc21=ccr[ccr.表.isin(["养殖馆+"])].reset_index(drop=True)
for i in ming:
    cc21=ccr[ccr.表.isin([i])].reset_index(drop=True)
    print(cc21)
for i in ming:
    cc21=ccr[ccr.表.isin([i])].reset_index(drop=True)
    X=cc21["结果"]
    outlier_label = model_isof.fit_predict(X)
    ccr[ccr.表.isin([i])]["outlier_pd"]=outlier_label
model_isof = IsolationForest(n_estimators=20)
from sklearn.ensemble import IsolationForest
model_isof = IsolationForest(n_estimators=20)
for i in ming:
    cc21=ccr[ccr.表.isin([i])].reset_index(drop=True)
    X=cc21["结果"]
    outlier_label = model_isof.fit_predict(X)
    ccr[ccr.表.isin([i])]["outlier_pd"]=outlier_label
for i in ming:
    cc21=ccr[ccr.表.isin([i])].reset_index(drop=True)
    X=np.array(cc21["结果"]).reshape(-1,1)
    outlier_label = model_isof.fit_predict(X)
    ccr[ccr.表.isin([i])]["outlier_pd"]=outlier_label
cc21=ccr[ccr.表.isin(["养殖馆+"])].reset_index(drop=True)
X=np.array(cc21["结果"]).reshape(-1,1)
outlier_label = model_isof.fit_predict(X)
ccr[ccr.表.isin(["养殖馆+"])]["outlier_pd"]=outlier_label

outlier_label.value_counts()
ccr["outlier_pd"]=outlier_label
for i in ming:
    cc21=ccr[ccr.表.isin([i])].reset_index(drop=True)
    X=np.array(cc21["结果"]).reshape(-1,1)
    outlier_label = model_isof.fit_predict(X)
    X["outlier_pd"]=outlier_label
cc21["outlier_pd"]=outlier_label
for i in ming:
    cc21=ccr[ccr.表.isin([i])].reset_index(drop=True)
    X=np.array(cc21["结果"]).reshape(-1,1)
    outlier_label = model_isof.fit_predict(X)
    cc21["outlier_pd"]=outlier_label
    cc22=pd.concat([cc22,cc21])
for i in ming:
    cc21=ccr[ccr.表.isin([i])].reset_index(drop=True)
    X=np.array(cc21["结果"]).reshape(-1,1)
    outlier_label = model_isof.fit_predict(X)
    cc21["outlier_pd"]=outlier_label
    cc22=pd.concat([cc21])
cc22=pd.DataFrame(columns=['表', '时间', '结果'])
for i in ming:
    cc21=ccr[ccr.表.isin([i])].reset_index(drop=True)
    X=np.array(cc21["结果"]).reshape(-1,1)
    outlier_label = model_isof.fit_predict(X)
    cc21["outlier_pd"]=outlier_label
    cc22=pd.concat([cc21,cc22])
cc22.to_csv("cc2.csv")
cc22.to_csv("cc22.csv")
cc1=pd.read_csv("附件_一季度.csv",encoding="gbk")
cc2=pd.read_csv("附件_二季度.csv",encoding="gbk")
cc3=pd.read_csv("附件_三季度.csv",encoding="gbk")
cc4=pd.read_csv("附件_四季度.csv",encoding="gbk")
df = pd.concat([cc1, cc2,cc3,cc4])
cc=df.loc[df["水表名"]=="校医院南+"]
bb=df.loc[df["水表名"]=="XXX校医院"]
aa=df.loc[df["水表名"]=="车队+"]
ff=pd.merge(cc,bb,on="采集时间")
gg=pd.merge(ff,aa,on="采集时间")
gg["差"]=gg["用量_x"]-gg["用量_y"]
len(gg["差"].loc[gg["差"]<0])
ccc=t403[t403.水表号.isin(["3620300300"])]
ccc.set_index(["采集时间2"], inplace=True)

cc=cc.groupby("采集时间2")["cha"].sum()

cccc=ccc["cha"]-cc
cccc.to_csv("cccc.csv")
cc=df.groupby(["水表名"])["用量"].sum()
cc
ming=ccr["表"].unique()
model_isof = IsolationForest(n_estimators=15)
#养殖馆+
cc22=pd.DataFrame(columns=['表', '时间', '结果'])
for i in ming:
    cc21=ccr[ccr.表.isin([i])].reset_index(drop=True)
    X=np.array(cc21["结果"]).reshape(-1,1)
    outlier_label = model_isof.fit_predict(X)
    cc21["outlier_pd"]=outlier_label
    cc22=pd.concat([cc21,cc22])

## ---(Sat Sep 19 21:41:29 2020)---
import pandas as pd 
cc1=pd.read_excel("疫情.xlsx")
"""
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("疫情.xlsx",encoding="gbk")
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("疫情.xlsx",encoding="gbk")
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("疫情.xlsx",encoding="gbk")
d1=pd.read_excel("疫情.xlsx",encoding="gbk")
"""
Created on Sat Sep 14 11:24:28 2019

@author: 92156
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("疫情.xlsx",encoding="gbk")
y=np.array(d1["时间"])
X=np.array(d1["疫情人数"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
#%画图 PM2.5 /15  00：14
plt.plot(rbf_svr_y_predict[:100,],c="r",label="NO2预测")
plt.plot(y_test[:100,],c="b",label='NO2')
plt.legend()
plt.ylabel("real and predicted value") 
plt.title("regression result comparison")
"""
Created on Sat Sep 14 11:24:28 2019

@author: 92156
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("疫情.xlsx",encoding="gbk")
y=np.array(d1["时间"])
X=np.array(d1["疫情人数"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
X_train=np.array(X_train).reshape(-1,1)
X_test=np.array(X_test).reshape(-1,1)
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
#%画图 PM2.5 /15  00：14
plt.plot(rbf_svr_y_predict[:100,],c="r",label="NO2预测")
plt.plot(y_test[:100,],c="b",label='NO2')
plt.legend()
plt.ylabel("real and predicted value") 
plt.title("regression result comparison")
reset
"""
Created on Sat Sep 14 11:24:28 2019

@author: 92156
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("疫情.xlsx",encoding="gbk")
y=np.array(d1["时间"])
X=np.array(d1["疫情人数"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
X_train=np.array(X_train).reshape(-1,1)
X_test=np.array(X_test).reshape(-1,1)
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
#%画图 PM2.5 /15  00：14
plt.plot(rbf_svr_y_predict[:100,],c="r",label="NO2预测")
plt.plot(y_test[:100,],c="b",label='NO2')
plt.legend()
plt.ylabel("real and predicted value") 
plt.title("regression result comparison")
"""
Created on Sat Sep 14 11:24:28 2019

@author: 92156
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("疫情.xlsx",encoding="gbk")
y=np.array(d1["时间"])
X=np.array(d1["疫情人数"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
X_train=np.array(X_train).reshape(-1,1)
X_test=np.array(X_test).reshape(-1,1)
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
#%画图 PM2.5 /15  00：14
plt.plot(rbf_svr_y_predict[:,],c="r",label="预测")
plt.plot(y_test[:,],c="b",label='实际')
plt.legend()
plt.ylabel("real and predicted value") 
plt.title("regression result comparison")
"""
Created on Sat Sep 14 11:24:28 2019

@author: 92156
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import  GridSearchCV
import matplotlib.pyplot as plt

d1=pd.read_excel("疫情.xlsx",encoding="gbk")
y=np.array(d1["时间"])
X=np.array(d1["疫情人数"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

ss_X = StandardScaler()
ss_y = StandardScaler()
X_train=np.array(X_train).reshape(-1,1)
X_test=np.array(X_test).reshape(-1,1)
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
from sklearn.svm import SVR



linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print ('R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test))
print ('The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print ('R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test))
print ('the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print ('the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print ('R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test))
print ('the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
#%画图 PM2.5 /15  00：14
plt.plot(rbf_svr_y_predict[:,],c="r",label="yuce")
plt.plot(y_test[:,],c="b",label='shiji')
plt.legend()
plt.ylabel("real and predicted value") 
plt.title("regression result comparison")

## ---(Fri Oct  2 15:49:38 2020)---
import smtplib
from email.mime.text import MIMEText
from email.header import Header

#发送html格式的邮件

#发送邮箱
sender="921560622@qq.com"
#接收邮箱
receiver="921560622@qq.com"
#发送邮件主题
subject="灵枢自动化测试"
#发送邮箱服务器
smtpserver="smtp.qq.com"
#发送邮箱用户/密码
username="921560622@qq.com"
password="chenqingbin//"

#HTML形式的邮件
msg=MIMEText("<html><h1>This Test Report!</h1></html>","html","utf-8")
msg["Subject"]=Header(subject,"utf-8")

smtp=smtplib.SMTP_SSL(smtpserver, 465)
smtp.login(username,password)
smtp.sendmail(sender,receiver,msg.as_string())
smtp.quit()
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.patches import Arc, Circle, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.patches import Arc, Circle, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties


length = 20  
R = 3**0.5*length/(3**0.5*cos(pi/12)-sin(pi/12))
r = 2*sin(pi/12)*R/3**0.5

arc1 = Arc([0, length], width=2*r, height=2*r, 
           angle=0, theta1=30, theta2=150, ec='orange', linewidth=4)  ##ec为线条颜色，可以自由替换

#arc1 = Arc([0, length], width=2*r, height=2*r, angle=0, theta1=30, theta2=150, ec='orange', linewidth=4)
arc2 = Arc([-length/2, length/2*3**0.5], width=2*r, height=2*r, 
           angle=0, theta1=60, theta2=180, ec='orange', linewidth=4)
arc3 = Arc([-length/2*3**0.5, length/2], width=2*r, height=2*r, 
           angle=0, theta1=90, theta2=210, ec='orange', linewidth=4)
arc4 = Arc([-length, 0], width=2*r, height=2*r, angle=0, theta1=120, theta2=240, ec='orange', linewidth=4)
arc5 = Arc([-length/2*3**0.5, -length/2], width=2*r, height=2*r, 
           angle=0, theta1=150, theta2=270, ec='orange', linewidth=4)
arc6 = Arc([-length/2, -length/2*3**0.5], width=2*r, height=2*r,
           angle=0, theta1=180, theta2=300, ec='orange', linewidth=4)
arc7 = Arc([0, -length], width=2*r, height=2*r, angle=0, theta1=210, theta2=330, ec='orange', linewidth=4)
arc8 = Arc([length/2, -length/2*3**0.5], width=2*r, height=2*r,
           angle=0, theta1=240, theta2=360, ec='orange', linewidth=4)
arc9 = Arc([length/2*3**0.5, -length/2], width=2*r, height=2*r,
           angle=0, theta1=270, theta2=390, ec='orange', linewidth=4)
arc10 = Arc([length, 0], width=2*r, height=2*r, angle=0, theta1=300, theta2=420, ec='orange', linewidth=4)
arc11 = Arc([length/2*3**0.5, length/2], width=2*r, height=2*r,
            angle=0, theta1=330, theta2=450, ec='orange', linewidth=4)
arc12 = Arc([length/2, length/2*3**0.5], width=2*r, height=2*r,
            angle=0, theta1=0, theta2=120, ec='orange', linewidth=4)


circle = Circle((0,0), R, ec='orange', fc='white', linewidth=4) ##ec为线条颜色，fc为填充颜色,可以自由替换

wedge1 = Wedge([-2, 2], R-5, 90, 180,
               ec='orange', fc=r'white', linewidth=4) ##ec为线条颜色，fc为填充颜色,可以自由替换


wedge2 = Wedge([-5, 5], R-12, 90, 180, ec='orange',
               fc=r'white', linewidth=4)
wedge3 = Wedge([-2, -2], R-5, 180, 270, ec='orange', 
               fc=r'white', linewidth=4)
wedge4 = Wedge([-5, -5], R-12, 180, 270, ec='orange', 
               fc=r'white', linewidth=4)
wedge5 = Wedge([2, -2], R-5, 270, 360, ec='orange', 
               fc=r'white', linewidth=4)
wedge6 = Wedge([5, -5], R-12, 270, 360, ec='orange',
               fc=r'white', linewidth=4)
wedge7 = Wedge([2, 2], R-5, 0, 90, ec='orange', 
               fc=r'white', linewidth=4)
wedge8 = Wedge([5, 5], R-12, 0, 90, ec='orange',
               fc=r'white', linewidth=4)


art_list = [arc1, arc2, arc3, arc4, arc5, arc6, arc7, arc8, arc9, arc10, arc11, arc12]
art_list.extend([circle, wedge1, wedge2, wedge3, wedge4, wedge5, wedge6, wedge7, wedge8])
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')
for a in art_list:
    ax.add_patch(a)


plt.axis('off')
font_set = FontProperties(fname=r"Alibaba-PuHuiTi-Medium.ttf", size=12) ##可以自由下载字体使用
plt.text(-15, -2.5, 'chaizi', bbox=dict(boxstyle='square', fc="w", ec='orange', linewidth=4),  fontsize=50, color='orange') ##ec为线条颜色，color为字体颜色,可以自由替换
plt.text(-28, -33, 'Python画月饼，千里共禅娟',fontproperties=font_set, fontsize=30, color='#aa4a30')
plt.ylim([-35, 35])
plt.xlim([-35, 35])

plt.show()
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.patches import Arc, Circle, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties


length = 20  
R = 3**0.5*length/(3**0.5*cos(pi/12)-sin(pi/12))
r = 2*sin(pi/12)*R/3**0.5

arc1 = Arc([0, length], width=2*r, height=2*r, 
           angle=0, theta1=30, theta2=150, ec='orange', linewidth=4)  ##ec为线条颜色，可以自由替换

#arc1 = Arc([0, length], width=2*r, height=2*r, angle=0, theta1=30, theta2=150, ec='orange', linewidth=4)
arc2 = Arc([-length/2, length/2*3**0.5], width=2*r, height=2*r, 
           angle=0, theta1=60, theta2=180, ec='orange', linewidth=4)
arc3 = Arc([-length/2*3**0.5, length/2], width=2*r, height=2*r, 
           angle=0, theta1=90, theta2=210, ec='orange', linewidth=4)
arc4 = Arc([-length, 0], width=2*r, height=2*r, angle=0, theta1=120, theta2=240, ec='orange', linewidth=4)
arc5 = Arc([-length/2*3**0.5, -length/2], width=2*r, height=2*r, 
           angle=0, theta1=150, theta2=270, ec='orange', linewidth=4)
arc6 = Arc([-length/2, -length/2*3**0.5], width=2*r, height=2*r,
           angle=0, theta1=180, theta2=300, ec='orange', linewidth=4)
arc7 = Arc([0, -length], width=2*r, height=2*r, angle=0, theta1=210, theta2=330, ec='orange', linewidth=4)
arc8 = Arc([length/2, -length/2*3**0.5], width=2*r, height=2*r,
           angle=0, theta1=240, theta2=360, ec='orange', linewidth=4)
arc9 = Arc([length/2*3**0.5, -length/2], width=2*r, height=2*r,
           angle=0, theta1=270, theta2=390, ec='orange', linewidth=4)
arc10 = Arc([length, 0], width=2*r, height=2*r, angle=0, theta1=300, theta2=420, ec='orange', linewidth=4)
arc11 = Arc([length/2*3**0.5, length/2], width=2*r, height=2*r,
            angle=0, theta1=330, theta2=450, ec='orange', linewidth=4)
arc12 = Arc([length/2, length/2*3**0.5], width=2*r, height=2*r,
            angle=0, theta1=0, theta2=120, ec='orange', linewidth=4)


circle = Circle((0,0), R, ec='orange', fc='white', linewidth=4) ##ec为线条颜色，fc为填充颜色,可以自由替换

wedge1 = Wedge([-2, 2], R-5, 90, 180,
               ec='orange', fc=r'white', linewidth=4) ##ec为线条颜色，fc为填充颜色,可以自由替换


wedge2 = Wedge([-5, 5], R-12, 90, 180, ec='orange',
               fc=r'white', linewidth=4)
wedge3 = Wedge([-2, -2], R-5, 180, 270, ec='orange', 
               fc=r'white', linewidth=4)
wedge4 = Wedge([-5, -5], R-12, 180, 270, ec='orange', 
               fc=r'white', linewidth=4)
wedge5 = Wedge([2, -2], R-5, 270, 360, ec='orange', 
               fc=r'white', linewidth=4)
wedge6 = Wedge([5, -5], R-12, 270, 360, ec='orange',
               fc=r'white', linewidth=4)
wedge7 = Wedge([2, 2], R-5, 0, 90, ec='orange', 
               fc=r'white', linewidth=4)
wedge8 = Wedge([5, 5], R-12, 0, 90, ec='orange',
               fc=r'white', linewidth=4)


art_list = [arc1, arc2, arc3, arc4, arc5, arc6, arc7, arc8, arc9, arc10, arc11, arc12]
art_list.extend([circle, wedge1, wedge2, wedge3, wedge4, wedge5, wedge6, wedge7, wedge8])
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')
for a in art_list:
    ax.add_patch(a)


plt.axis('off')
font_set = FontProperties( size=12) ##可以自由下载字体使用
plt.text(-15, -2.5, 'chaizi', bbox=dict(boxstyle='square', fc="w", ec='orange', linewidth=4),  fontsize=50, color='orange') ##ec为线条颜色，color为字体颜色,可以自由替换
plt.text(-28, -33, 'Python画月饼，千里共禅娟',fontproperties=font_set, fontsize=30, color='#aa4a30')
plt.ylim([-35, 35])
plt.xlim([-35, 35])

plt.show()
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.patches import Arc, Circle, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.patches import Arc, Circle, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties


length = 20  
R = 3**0.5*length/(3**0.5*cos(pi/12)-sin(pi/12))
r = 2*sin(pi/12)*R/3**0.5

arc1 = Arc([0, length], width=2*r, height=2*r, 
           angle=0, theta1=30, theta2=150, ec='orange', linewidth=4)  ##ec为线条颜色，可以自由替换

#arc1 = Arc([0, length], width=2*r, height=2*r, angle=0, theta1=30, theta2=150, ec='orange', linewidth=4)
arc2 = Arc([-length/2, length/2*3**0.5], width=2*r, height=2*r, 
           angle=0, theta1=60, theta2=180, ec='orange', linewidth=4)
arc3 = Arc([-length/2*3**0.5, length/2], width=2*r, height=2*r, 
           angle=0, theta1=90, theta2=210, ec='orange', linewidth=4)
arc4 = Arc([-length, 0], width=2*r, height=2*r, angle=0, theta1=120, theta2=240, ec='orange', linewidth=4)
arc5 = Arc([-length/2*3**0.5, -length/2], width=2*r, height=2*r, 
           angle=0, theta1=150, theta2=270, ec='orange', linewidth=4)
arc6 = Arc([-length/2, -length/2*3**0.5], width=2*r, height=2*r,
           angle=0, theta1=180, theta2=300, ec='orange', linewidth=4)
arc7 = Arc([0, -length], width=2*r, height=2*r, angle=0, theta1=210, theta2=330, ec='orange', linewidth=4)
arc8 = Arc([length/2, -length/2*3**0.5], width=2*r, height=2*r,
           angle=0, theta1=240, theta2=360, ec='orange', linewidth=4)
arc9 = Arc([length/2*3**0.5, -length/2], width=2*r, height=2*r,
           angle=0, theta1=270, theta2=390, ec='orange', linewidth=4)
arc10 = Arc([length, 0], width=2*r, height=2*r, angle=0, theta1=300, theta2=420, ec='orange', linewidth=4)
arc11 = Arc([length/2*3**0.5, length/2], width=2*r, height=2*r,
            angle=0, theta1=330, theta2=450, ec='orange', linewidth=4)
arc12 = Arc([length/2, length/2*3**0.5], width=2*r, height=2*r,
            angle=0, theta1=0, theta2=120, ec='orange', linewidth=4)


circle = Circle((0,0), R, ec='orange', fc='white', linewidth=4) ##ec为线条颜色，fc为填充颜色,可以自由替换

wedge1 = Wedge([-2, 2], R-5, 90, 180,
               ec='orange', fc=r'white', linewidth=4) ##ec为线条颜色，fc为填充颜色,可以自由替换


wedge2 = Wedge([-5, 5], R-12, 90, 180, ec='orange',
               fc=r'white', linewidth=4)
wedge3 = Wedge([-2, -2], R-5, 180, 270, ec='orange', 
               fc=r'white', linewidth=4)
wedge4 = Wedge([-5, -5], R-12, 180, 270, ec='orange', 
               fc=r'white', linewidth=4)
wedge5 = Wedge([2, -2], R-5, 270, 360, ec='orange', 
               fc=r'white', linewidth=4)
wedge6 = Wedge([5, -5], R-12, 270, 360, ec='orange',
               fc=r'white', linewidth=4)
wedge7 = Wedge([2, 2], R-5, 0, 90, ec='orange', 
               fc=r'white', linewidth=4)
wedge8 = Wedge([5, 5], R-12, 0, 90, ec='orange',
               fc=r'white', linewidth=4)


art_list = [arc1, arc2, arc3, arc4, arc5, arc6, arc7, arc8, arc9, arc10, arc11, arc12]
art_list.extend([circle, wedge1, wedge2, wedge3, wedge4, wedge5, wedge6, wedge7, wedge8])
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')
for a in art_list:
    ax.add_patch(a)


plt.axis('off')
font_set = FontProperties( size=12) ##可以自由下载字体使用
plt.text(-15, -2.5, 'chaizi', bbox=dict(boxstyle='square', fc="w", ec='orange', linewidth=4),  fontsize=50, color='orange') ##ec为线条颜色，color为字体颜色,可以自由替换
plt.text(-28, -33, 'Python画月饼，千里共禅娟',fontproperties=font_set, fontsize=30, color='#aa4a30')
plt.ylim([-35, 35])
plt.xlim([-35, 35])

plt.show()
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.patches import Arc, Circle, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties


length = 20  
R = 3**0.5*length/(3**0.5*cos(pi/12)-sin(pi/12))
r = 2*sin(pi/12)*R/3**0.5

arc1 = Arc([0, length], width=2*r, height=2*r, 
           angle=0, theta1=30, theta2=150, ec='orange', linewidth=4)  ##ec为线条颜色，可以自由替换

#arc1 = Arc([0, length], width=2*r, height=2*r, angle=0, theta1=30, theta2=150, ec='orange', linewidth=4)
arc2 = Arc([-length/2, length/2*3**0.5], width=2*r, height=2*r, 
           angle=0, theta1=60, theta2=180, ec='orange', linewidth=4)
arc3 = Arc([-length/2*3**0.5, length/2], width=2*r, height=2*r, 
           angle=0, theta1=90, theta2=210, ec='orange', linewidth=4)
arc4 = Arc([-length, 0], width=2*r, height=2*r, angle=0, theta1=120, theta2=240, ec='orange', linewidth=4)
arc5 = Arc([-length/2*3**0.5, -length/2], width=2*r, height=2*r, 
           angle=0, theta1=150, theta2=270, ec='orange', linewidth=4)
arc6 = Arc([-length/2, -length/2*3**0.5], width=2*r, height=2*r,
           angle=0, theta1=180, theta2=300, ec='orange', linewidth=4)
arc7 = Arc([0, -length], width=2*r, height=2*r, angle=0, theta1=210, theta2=330, ec='orange', linewidth=4)
arc8 = Arc([length/2, -length/2*3**0.5], width=2*r, height=2*r,
           angle=0, theta1=240, theta2=360, ec='orange', linewidth=4)
arc9 = Arc([length/2*3**0.5, -length/2], width=2*r, height=2*r,
           angle=0, theta1=270, theta2=390, ec='orange', linewidth=4)
arc10 = Arc([length, 0], width=2*r, height=2*r, angle=0, theta1=300, theta2=420, ec='orange', linewidth=4)
arc11 = Arc([length/2*3**0.5, length/2], width=2*r, height=2*r,
            angle=0, theta1=330, theta2=450, ec='orange', linewidth=4)
arc12 = Arc([length/2, length/2*3**0.5], width=2*r, height=2*r,
            angle=0, theta1=0, theta2=120, ec='orange', linewidth=4)


circle = Circle((0,0), R, ec='orange', fc='white', linewidth=4) ##ec为线条颜色，fc为填充颜色,可以自由替换

wedge1 = Wedge([-2, 2], R-5, 90, 180,
               ec='orange', fc=r'white', linewidth=4) ##ec为线条颜色，fc为填充颜色,可以自由替换


wedge2 = Wedge([-5, 5], R-12, 90, 180, ec='orange',
               fc=r'white', linewidth=4)
wedge3 = Wedge([-2, -2], R-5, 180, 270, ec='orange', 
               fc=r'white', linewidth=4)
wedge4 = Wedge([-5, -5], R-12, 180, 270, ec='orange', 
               fc=r'white', linewidth=4)
wedge5 = Wedge([2, -2], R-5, 270, 360, ec='orange', 
               fc=r'white', linewidth=4)
wedge6 = Wedge([5, -5], R-12, 270, 360, ec='orange',
               fc=r'white', linewidth=4)
wedge7 = Wedge([2, 2], R-5, 0, 90, ec='orange', 
               fc=r'white', linewidth=4)
wedge8 = Wedge([5, 5], R-12, 0, 90, ec='orange',
               fc=r'white', linewidth=4)


art_list = [arc1, arc2, arc3, arc4, arc5, arc6, arc7, arc8, arc9, arc10, arc11, arc12]
art_list.extend([circle, wedge1, wedge2, wedge3, wedge4, wedge5, wedge6, wedge7, wedge8])
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')
for a in art_list:
    ax.add_patch(a)


plt.axis('off')
font_set = FontProperties( size=12) ##可以自由下载字体使用
plt.text(-15, -2.5, 'chaizi', bbox=dict(boxstyle='square', fc="w", ec='orange', linewidth=4),  fontsize=50, color='orange') ##ec为线条颜色，color为字体颜色,可以自由替换
plt.ylim([-35, 35])
plt.xlim([-35, 35])

plt.show()

## ---(Sat Oct 24 19:50:41 2020)---
"""
Created on Sat Oct 24 19:51:29 2020

@author: 92156
"""

import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import  stats
import warnings
warnings.filterwarnings("ignore")
import sys

train_data_file="D:\天池数据\工业预测\zhengqi_train.txt"
text_data_file="D:\天池数据\工业预测\zhengqi_test.csv"
train_data=pd.read_csv(train_data_file,sep="\t",encoding="unicode_escape")
text_data=pd.read_csv(text_data_file,encoding="gbk")
"""
Created on Sat Oct 24 19:51:29 2020

@author: 92156
"""

import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import  stats
import warnings
warnings.filterwarnings("ignore")
import sys

train_data_file="D:\天池数据\工业预测\zhengqi_train.txt"
text_data_file="D:\天池数据\工业预测\zhengqi_test.csv"
train_data=pd.read_csv(train_data_file,sep="\t",encoding="unicode_escape")
text_data=pd.read_csv(text_data_file,encoding="gbk")


text_data.info()
train_data.info()


train_data.describe()


text_data.describe()

train_data.head()
text_data.head()

fig=plt.figure(figsize=(5,9))
sns.boxplot(train_data["V0"],orient="v",width=0.5)
plt.show()

column=train_data.columns.tolist()[:39]
fig=plt.figure(figsize=(80,60),dpi=75)
for i in range(38):
    plt.subplot(7,8,i+1) #13行3列子图
    sns.boxplot(train_data[column[i]],orient="v",width=0.5) # 箱形图
    plt.ylabel(column[i],fontsize=36)

plt.show()


#function to detect outliers based on the predictions of model 
def find_outliers(model,X,y,sigma=3):
    #predict. y value using model 
    try: 
        y_pred=pd.Series(model.predict(X),index=y.index)
    #if predicting fails ,try fitting the model first \
    except:
        model.fit(X,y)
        y_pred=pd.Series(model.predict(X),index=y.index)
    #calculat residuale betweenw the model prediction and true y values
    resid=y-y_pred
    mean_resid=resid.mean()
    std_resid=resid.std()
    #calculate z statistic , define outliers to be where |z| > sigma
    z=(resid-mean_resid)/std_resid
    outliers=z[abs(z)>sigma].index
    
    #print and plot the results 
    
    print("R2=",model.score(X,y))
    print("msc=",mean_squared_error(y,y_pred))
    print("------------------------")
    
    print("mean of residuals:",mean_resid)
    print("std of residuals :",std_resid)
    print("------------------------")
    
    
    print(len(outliers),"outliers:")
    print(outliers.tolist())
    
    plt.figure(figsize=(15,5))
    ax_131=plt.subplot(1,3,1)
    plt.plot(y.loc[outliers],y_pred.loc[outliers],"ro")
    plt.legend(["Accepted","Outlier"])
    plt.xlabel("y")
    plt.ylabel("y_pred")
    
    ax_132=plt.subplot(1,3,2)
    plt.plot(y,y-y_pred,".")
    plt.plot(y.loc[outliers],y.loc[outliers]-y_pred.loc[outliers],"ro")
    plt.legend(["Accepted","Outlier"])
    plt.xlabel("y")
    plt.ylabel("y-y_pred")
    
    ax_133=plt.subplot(1,3,3)
    z.plot.hist(bins=50,ax=ax_133)
    z.loc[outliers].plot.hist(color="r",bins=50,ax=ax_133)
    plt.legend(["Accepted","Outlier"])
    plt.xlabel("z")
    
    return outliers



from sklearn.linear_model import  Ridge
from sklearn.metrics import  mean_squared_error
X_train=train_data.iloc[:,0:-1]
y_train=train_data.iloc[:,-1]
outliers=find_outliers(Ridge(),X_train,y_train)

## ---(Tue Apr 13 12:52:35 2021)---
runfile('C:/Users/92156/Documents/GitHub/Boss_zhipin_spider/www_zhipin_com/www_zhipin_com/spiders/boss_zhipin_spider.py', wdir='C:/Users/92156/Documents/GitHub/Boss_zhipin_spider/www_zhipin_com/www_zhipin_com/spiders')
import json
import time
import winsound

import requests
import scrapy

from www_zhipin_com.items import WwwZhipinComItem


class ZhipinSpider(scrapy.Spider):
    
    handle_httpstatus_list = [302]
    
    name = 'zhipin'
    
    allowed_domains = ['www.zhipin.com']
    
    start_urls = [
        "http://www.zhipin.com/",
    ]
    
    # positionUrl = 'https://www.zhipin.com/c101050100/?query=python'
    positionUrl = 'https://www.zhipin.com/'
    # 当前省份的下标
    currentProv = 0
    # 当前页码
    currentPage = 1
    # 当前城市的下标
    currentCity = 0
    
    cityListUrl = "https://www.zhipin.com/common/data/city.json"
    
    cityList = []
    
    headers = {
        'x-devtools-emulate-network-conditions-client-id': "5f2fc4da-c727-43c0-aad4-37fce8e3ff39",
        'upgrade-insecure-requests': "1",
        'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36",
        'accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        'dnt': "1",
        'accept-encoding': "gzip, deflate",
        'accept-language': "zh-CN,zh;q=0.8,en;q=0.6",
        'cookie': "__c=1501326829; lastCity=101020100; __g=-; __l=r=https%3A%2F%2Fwww.google.com.hk%2F&l=%2F; __a=38940428.1501326829..1501326829.20.1.20.20; Hm_lvt_194df3105ad7148dcf2b98a91b5e727a=1501326839; Hm_lpvt_194df3105ad7148dcf2b98a91b5e727a=1502948718; __c=1501326829; lastCity=101020100; __g=-; Hm_lvt_194df3105ad7148dcf2b98a91b5e727a=1501326839; Hm_lpvt_194df3105ad7148dcf2b98a91b5e727a=1502954829; __l=r=https%3A%2F%2Fwww.google.com.hk%2F&l=%2F; __a=38940428.1501326829..1501326829.21.1.21.21",
        'cache-control': "no-cache",
        'postman-token': "76554687-c4df-0c17-7cc0-5bf3845c9831"
    }
    
    def parse(self, response):
        
        print(response.status)
        if response.status == 302:
            winsound.MessageBeep()
            # 等待用户输入验证码
            input('please input verify code to continue:')
            # self.crawler.engine.close_spider(self, 'done!' % response.text)
        print("request->" + response.url)
        is_one_page = response.css('div.job-list>div.page').extract()
        is_end = response.css(
            'div.job-list>div.page>a[class*="next disabled"]::attr(class)').extract()
        job_list = response.css('div.job-list>ul>li')
        for job in job_list:
            # 数据获取
            item = WwwZhipinComItem()
            # job_primary = job.css('div.job-primary')
            item['pid'] = job.css(
                'div.info-primary>h3>a::attr(data-jid)').extract_first().strip()
            item['positionName'] = job.css(
                'div.job-title::text').extract_first().strip()
            item['salary'] = job.css(
                'div.info-primary>h3>a> span::text').extract_first().strip()
            
            info_primary = job.css('div.info-primary>p::text').extract()
            item['city'] = info_primary[0].strip()
            item['workYear'] = info_primary[1].strip()
            item['education'] = info_primary[2].strip()
            
            item['companyShortName'] = job.css(
                'div.company-text>h3>a::text').extract_first().strip()
            company_info = job.css('div.company-text>p::text').extract()
            if len(company_info) == 3:
                item['industryField'] = company_info[0].strip()
                item['financeStage'] = company_info[1].strip()
                item['companySize'] = company_info[2].strip()
            
            item['time'] = job.css(
                'div.info-publis>p::text').extract_first().strip()
            interviewer_info = job.css('div.info-publis>h3::text').extract()
            item['interviewer'] = interviewer_info[1]
            
            item['updated_at'] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime())
            yield item
        
        # 下一页不可点击则表示到底，退出
        # print(len(is_end))
        # print(len(is_one_page))
        if len(is_end) != 0 or len(is_one_page) == 0:
            # self.crawler.engine.close_spider(self, 'done!' % response.text)
            #     todo: 城市id变化，是否变化传入next_request的参数中，预先导入城市列表，然后循环
            self.currentCity += 1
            self.currentPage = 0
            prov_index = self.currentProv
            # 跨省
            # print(len(self.cityList[prov_index]['subLevelModelList']))
            if self.currentCity >= len(self.cityList[prov_index]['subLevelModelList']):
                self.currentProv += 1
                self.currentCity = 0
                self.currentPage = 0
        
        if self.currentProv == 34:
            self.crawler.engine.close_spider(self, 'done!' % response.text)
        # 翻页
        self.currentPage += 1
        time.sleep(2)
        yield self.next_request(self.currentProv, self.currentCity)
    
    def start_requests(self):
        # start_requests 只调用一次,初始化时获取city列表
        res = requests.get(self.cityListUrl, headers=self.headers).content
        city = json.loads(res)
        
        # 调试用
        self.cityList = city['data']['cityList']
        
        return [self.next_request(self.currentProv, self.currentCity)]
    
    def next_request(self, current_prov, current_city):
        print("current_prov"+str(current_prov))
        print("current_city"+str(current_city))
        cur_city_id = 'c' + \
            str(self.cityList[current_prov]
                ['subLevelModelList'][current_city]['code'])
        print(cur_city_id)
        # 这里url写想要查找什么职业
        return scrapy.http.FormRequest(
            self.positionUrl + cur_city_id + (
                "?query=python&page=%d&ka=page-%d" % (self.currentPage, self.currentPage)),
            headers=self.headers,
            callback=self.parse)
runfile('C:/Users/92156/Documents/GitHub/Boss_zhipin_spider/www_zhipin_com/www_zhipin_com/spiders/boss_zhipin_spider.py', wdir='C:/Users/92156/Documents/GitHub/Boss_zhipin_spider/www_zhipin_com/www_zhipin_com/spiders')
import json
import time
import winsound

import requests
import scrapy

from www_zhipin_com.items import WwwZhipinComItem