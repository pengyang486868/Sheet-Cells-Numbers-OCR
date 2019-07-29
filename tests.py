from core import CellRecognizer
import numpy as np
from PIL import Image
import os


def test1():
    # path = r'D:\pic\splits'
    path = r'D:\pic'
    r = CellRecognizer(7)

    im = Image.open(os.path.join(path, 'simple.jpg'))
    # result = r.ocr(im)
    result = r.ocr(Image.fromarray(np.array(im)[:, :, 0]))
    print(result)


def test2():
    sn = None
    a = np.array(sn, dtype=float)
    pass


test2()
