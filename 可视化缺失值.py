# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:11:22 2019

@author: 92156
"""

import numpy as ny
import pandas as pd
data=pd.read_excel("工作簿1(2).xlsx")
import missingno as msno
msno.matrix(data, labels=True)