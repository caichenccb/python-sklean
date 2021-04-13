

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
 
n=1
 
 
def save_to_excel(soup):
    for i in range(1,16):
        list = soup.find(class_='content-wrap').find_all(i)
 
    for item in list:
        item_name = item.find(href="").string
 
        global n
 
        sheet.write(n, 0, item_name)
 
        n = n + 1
 
def main(): 
    url = 'https://bbs.5g-yun.com/yuanma/shangcheng/'
    html = request_douban(url)
    soup = BeautifulSoup(html, 'lxml')
    save_to_excel(soup)
 
 
if __name__ == '__main__':
 
    main()
 
book.save(u'豆瓣最受欢迎的250部电影.xlsx')