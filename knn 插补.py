# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:44:04 2019

@author: 92156
"""


from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
 
def knn_missing_filled(x_train, y_train, test, k = 3, dispersed = True):
    if dispersed:
        clf = KNeighborsClassifier(n_neighbors = k, weights = "distance")
    else:
        clf = KNeighborsRegressor(n_neighbors = k, weights = "distance")
    
    clf.fit(x_train, y_train)
    return test.index, clf.predict(test)