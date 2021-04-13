# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 20:23:53 2019

@author: 92156
"""
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MulitinomialNB
GaussianNB()   #高斯朴素贝叶斯
#离散 简单 有些特征可能是连续型变量，比如说人的身高，物体的长度，这些特征可以转换成离散型的值
#，比如如果身高在160cm以下，特征值为1；在160cm和170cm之间，特征值为2；在170cm之上，
#特征值为3。也可以这样转换，将身高转换为3个特征，分别是f1、f2、f3，如果身高是160cm以下，
#这三个特征的值分别是1、0、0，若身高在170cm之上，这三个特征的值分别是0、0、1。不过这些方式都不够细腻，
#高斯模型可以解决这个问题。高斯模型假设这些一个特征的所有属于某个类别的观测值符合高斯分布，也就是：


BernoulliNB()    # 与MulitinomialNB相似   
#伯努利模型中，对于一个样本来说，其特征用的是全局的特征。


#MultinomialNB以出现的次数为特征值，BernoulliNB为二进制或布尔型特性
MulitinomialNB()  #离散  多项式模型  
#该模型常用于文本分类，特征是单词，值是单词的出现次数。


