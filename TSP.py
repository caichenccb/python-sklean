# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 22:46:02 2019

@author: 92156
"""
#先运行def的



#遗传算法
## 环境设置
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
N_POP = 10000
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

#最邻近算法


tour,tourDist = repNNTSP(cities)
print('rennTSP寻找到的最优路径为：' + str(tour))
print('nnTSP寻找到最优路径长度为：' + str(tourDist))
plotTour(tour, cities)
repnnTSPtour=tour

#优化最邻近算法
optimizedRoute, minDistance = opt(cityDist, tour)
print('nnTSP + 2OPT优化后的最优路径为：' + str(optimizedRoute))
print('nnTSP + 2OPT优化后的最优路径长度为：' + str(minDistance))
plotTour(optimizedRoute, cities)
repnnTSPtourOptimized, repnnTSPtourDistOptimized = opt(cityDist, repnnTSPtour, 2)
print('repnnTSPtour + 2OPT优化后的最优路径为：' + str(repnnTSPtourOptimized))
print('repnnTSPtour + 2OPT优化后的最优路径长度为：' + str(repnnTSPtourDistOptimized))
plotTour(repnnTSPtourOptimized, cities)
