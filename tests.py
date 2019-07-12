from core import CellRecognizer
import numpy as np
from PIL import Image
import os

path = r'D:\pic\splits'
r = CellRecognizer(2)

im = Image.open(os.path.join(path, '6-11.jpg'))
result = r.ocr(im)
print(result)
