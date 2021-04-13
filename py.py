
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
