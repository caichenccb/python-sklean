# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:40:10 2019

@author: 92156
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:02:41 2019

@author: 92156
"""

import requests
from bs4 import BeautifulSoup
import xlwt
 
 
 

def request_douban(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        return None
 
 
book=xlwt.Workbook(encoding='utf-8',style_compression=0)
 
sheet=book.add_sheet('豆瓣电影Top250',cell_overwrite_ok=True)
sheet.write(0,0,'名称')
sheet.write(0,1,'图片')
sheet.write(0,2,'排名')
sheet.write(0,3,'评分')
sheet.write(0,4,'作者')
sheet.write(0,5,'简介')
 
n=1
 
 
def save_to_excel(soup):
    list = soup.find(class_='grid_view').find_all('li')
 
    for item in list:
        item_name = item.find(class_='title').string
        item_img = item.find('a').find('img').get('src')
        item_index = item.find(class_='').string
        item_score = item.find(class_='rating_num').string
        item_author = item.find('p').text
        if(item.find(class_='inq')!=None):
            item_intr = item.find(class_='inq').string
 
        # print('爬取电影：' + item_index + ' | ' + item_name +' | ' + item_img +' | ' + item_score +' | ' + item_author +' | ' + item_intr )
        print('爬取电影：' + item_index + ' | ' + item_name  +' | ' + item_score  +' | ' + item_intr )
 
        global n
 
        sheet.write(n, 0, item_name)
        sheet.write(n, 1, item_img)
        sheet.write(n, 2, item_index)
        sheet.write(n, 3, item_score)
        sheet.write(n, 4, item_author)
        sheet.write(n, 5, item_intr)
 
        n = n + 1
 
 
def main(page):
    url = 'https://movie.douban.com/top250?start='+ str(page*25)+'&filter='
    html = request_douban(url)
    soup = BeautifulSoup(html, 'lxml')
    save_to_excel(soup)
 

if __name__ == '__main__':
 
    
    import threading
    import time
    class MyThread(threading.Thread):
      def __init__(self,threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    
      def run(self):
        for i in range(10):
            
            print("开始线程：" + self.name)
            moyu_time(self.name, self.counter, self.threadID,10)
            self.threadID +=2
            print("退出线程：" + self.name)
    
    def moyu_time(threadName, delay, page,ss):
      while ss:
        time.sleep(delay)
        main(page)
        ss -= 1
    
    
    # 创建新线程
    # 小帅b找了两个人来摸鱼
    # 让小明摸一次鱼休息1秒钟
    # 让小红摸一次鱼休息2秒钟
    thread1 = MyThread(1, "小明", 1)
    thread2 = MyThread(2, "小红", 1)
    
    # 开启新线程
    thread1.start()
    thread2.start()
    # 等待至线程中止
    thread1.join()
    thread2.join()
    print ("退出主线程")