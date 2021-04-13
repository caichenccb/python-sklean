# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:44:53 2019

@author: 92156
"""


from bs4 import BeautifulSoup
import requests
import time
import json
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
soup = BeautifulSoup(html_doc,'lxml')
 
print(soup.title.string)
#学习python的正确姿势
print(soup.title.parent.name)
#head
print(soup.find_all("p"))