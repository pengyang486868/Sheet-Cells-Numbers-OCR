import numpy as np
import config
import urllib.parse
import urllib.request as rq
import json
import cv2


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


# 透视变换
def ptrans(orgpath, tarpath, points):
    im = cv2.imread(orgpath, cv2.IMREAD_GRAYSCALE)
    tarw = np.float32(1.1 * max(points[1][0] - points[0][0], points[3][0] - points[2][0]))
    tarh = np.float32(1.1 * max(points[3][1] - points[0][1], points[2][1] - points[1][1]))
    canvasp = np.float32([[0, 0], [tarw, 0], [tarw, tarh], [0, tarh]])
    pmatrix = cv2.getPerspectiveTransform(points, canvasp)
    pimg = cv2.warpPerspective(im, pmatrix, (tarw, tarh))
    cv2.imwrite(tarpath, pimg)
