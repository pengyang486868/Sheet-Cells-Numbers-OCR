# coding: utf-8
from flask import Flask, jsonify, request
import os
from ocrutils import ptrans
import numpy as np
import uuid
import json
from core import SheetRecognizer

app = Flask(__name__)


#
@app.route('/test')
def test():
    httpargs = request.args
    return jsonify({'a': httpargs['a'], 'b': httpargs['b']})


#
@app.route('/testsheet')
def testsheet():
    path = r'D:\pic'
    name = 'cut2.jpg'
    r = SheetRecognizer(formats=3)
    result = r.ocr(os.path.join(path, name), rows=14, cols=12, stroke=0)
    np.savetxt(os.path.join(path, name + '-result.csv'), result, delimiter=',', fmt='%s')
    return ""


@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    splitnarr = None
    httpargs = None

    if request.method == 'POST':
        data = request.get_data()
        httpargs = json.loads(data.decode())
        splitnarr = httpargs['snumber']

    if request.method == 'GET':
        httpargs = request.args

    opath = httpargs['path']
    pts = np.float32([[httpargs['w1'], httpargs['h1']],
                      [httpargs['w2'], httpargs['h2']],
                      [httpargs['w3'], httpargs['h3']],
                      [httpargs['w4'], httpargs['h4']]])
    givenrows = int(httpargs['rows'])
    givencols = int(httpargs['cols'])
    givenstroke = int(httpargs['stroke'])

    newpath = os.path.join(path, 'temp', 'temp' + str(uuid.uuid1()) + '.jpg')
    ptrans(opath, newpath, pts)

    r = shtocr(sess, x, y, keep, newpath, splitn=splitnarr,
               rows=givenrows, cols=givencols, stroke=givenstroke, save=True)
    np.savetxt(opath + '-result.csv', r, delimiter=',', fmt='%s')
    return ""


# 启动程序
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9099)
