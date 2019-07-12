import numpy as np
import config
import urllib.parse
import urllib.request as rq
import json


# 识别结果转换 如[4,8] --> 48
def resultstr(arr):
    s = ''
    for i in arr:
        s += str(i)
    return s


# 调用网络
def netoutput(images, keep=0.5):
    data = json.dumps({'img': images.tolist(), 'keep': keep}).encode("utf-8")
    requrl = config.NET_ADDR + r'/single'
    req = rq.Request(url=requrl, data=data)
    res_data = rq.urlopen(req)
    cdata = str(res_data.read().decode())
    dataload = json.loads(cdata)
    return dataload['result']
