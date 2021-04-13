# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 23:11:29 2019

@author: 92156
"""

def compute_mse(b, w1, w2, x_data, y_data):
    """
    求均方差
    :param b: 截距
    :param w1: 斜率1
    :param w2: 斜率2
    :param x_data: 特征数据
    :param y_data: 标签数据
    """
    
    total_error = 0.0
    for i in range(0, len(x_data)):
        total_error += (y_data[i] - (b + w1 * x_data[i, 0] + w2 * x_data[i, 1])) ** 2
    return total_error / len(x_data)