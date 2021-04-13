# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:36:50 2019

@author: 92156
"""

import requests
import re
import json
 
def request_dandan(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        return None
 
 
def parse_result(html):
    pattern = re.compile("<h2>.*?href=(.*?)",re.S)
    items = re.findall(pattern,html)
    for item in items:
        yield {
            'range': item[0],
           
        }
 
 
def write_item_to_file(item):
    print('开始写入数据 ====> ' + str(item))
    with open('book.txt', 'a', encoding='UTF-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.close()
 
 
def main(page):
    url = "https://bbs.5g-yun.com/yuanma/shangcheng"
    html = request_dandan(url)
    items = parse_result(html) # 解析过滤我们想要的信息
    for item in items:
       print(item)
    for item in items:
        write_item_to_file(item)
 
 
if __name__ == "__main__":
    main()
