import numpy as np
from PIL import Image
import config
import os
import ocrutils


class SheetRecognizer:
    def __init__(self, formats, dosave=False, splpath=None, respath=None):
        self.recognizer = CellRecognizer(defaultfmt=3)
        self.formats = formats
        self.dosave = dosave
        self.splitpath = None
        self.resultpath = None
        if dosave:
            if not splpath:
                self.splitpath = config.SPLIT_PATH
            if not respath:
                self.resultpath = config.RESULT_PATH

    def ocr(self, path, rows, cols, splitn=None, stroke=10):
        im = Image.open(path)
        w, h = im.size
        wcell = w / cols
        hcell = h / rows

        # do wipe border and save for check
        im = self.wipeborder(im, wcell, hcell, w, h)
        if self.dosave:
            im.save(os.path.join(self.resultpath, 'wipe.jpg'))

        results = []
        # cutborder = 8
        cutborder = stroke
        if splitn is None:
            splitn = self.formats
        if isinstance(splitn, int):
            splitn = np.ones(rows) * splitn

        for i in range(rows):
            for j in range(cols):
                newimg = im.crop((
                    j * wcell + cutborder,
                    i * hcell,
                    (j + 1) * wcell - cutborder,
                    (i + 1) * hcell))
                if self.dosave:
                    newimg.save(os.path.join(self.splitpath, '%(i)d-%(j)d.jpg' % {'i': i, 'j': j}))
                # cur_result = resultstr(cellocr(newimg, splitn[i], sess, x, y, keep))
                cur_result = ocrutils.resultstr(self.recognizer.ocr(newimg, formatter=splitn[i]))
                # if not cur_result.any():
                #     continue
                if self.dosave:
                    newimg.save(os.path.join(self.resultpath, '%(i)d-%(j)d-(%(s)s).jpg'
                                             % {'s': cur_result, 'i': i, 'j': j}))
                results.append(cur_result)
                # print(i, j, cur_result)

        return np.array(results).reshape((rows, cols))

    # 擦除横纵表线
    @staticmethod
    def wipeborder(im, wcell, hcell, w, h):
        imarray = np.array(im)
        hlinewin = int(wcell * 1.2)
        vlinewin = int(hcell * 1.2)
        winrange = 1
        wipetol = 120

        for i in range(0, h - winrange):
            for j in range(0, w - hlinewin):
                for m in range(winrange):
                    if np.mean(imarray[i + m, j:j + hlinewin]) < wipetol:
                        imarray[i + m, j:j + hlinewin] = 255

        for i in range(0, h - vlinewin):
            for j in range(0, w - winrange):
                for m in range(winrange):
                    if np.mean(imarray[i:i + vlinewin, j + m]) < wipetol:
                        imarray[i:i + vlinewin, j + m] = 255

        im = Image.fromarray(imarray)
        return im


class CellRecognizer:
    def __init__(self, defaultfmt):
        # fixed format
        self.fmt = defaultfmt

    def ocr(self, im, formatter=None):
        if not formatter:
            ims = self.fig2images(im, self.fmt)
        else:
            ims = self.fig2images(im, formatter)
        if not ims.any():
            return np.array([])
        return ocrutils.netoutput(ims, keep=0.6)

    # 单元格按n格式切成标准的28*28子图
    @staticmethod
    def fig2images(im, n):
        tol = 0.9

        # should decide whether grey image or color image
        imraw = np.array(im)
        if imraw.ndim > 2:
            imraw = imraw[:, :, 0]

        imsolid = np.where(imraw >= 128, 1, 0)
        hnum = imsolid.shape[0]
        vnum = imsolid.shape[1]
        vvec = np.mean(imsolid, axis=0)
        hvec = np.mean(imsolid, axis=1)

        vvec_valley = np.where(vvec > tol)[0]
        vvec_peak = np.where(vvec < max(0.8, np.min(vvec) + 0.1))[0]

        valleys = []
        valley_start = 0
        for i in range(len(vvec_valley)):
            if i - valley_start != vvec_valley[i] - vvec_valley[valley_start]:
                valleys.append(int((vvec_valley[valley_start] + vvec_valley[i - 1]) / 2))
                valley_start = i

        if len(vvec_valley) > 0:
            valleys.append(int((vvec_valley[valley_start] + vvec_valley[-1]) / 2))

        peaks = []
        peak_start = 0
        for i in range(len(vvec_peak)):
            if i - peak_start != vvec_peak[i] - vvec_peak[peak_start]:
                peaks.append(int((vvec_peak[peak_start] + vvec_peak[i - 1]) / 2))
                peak_start = i

        if len(vvec_peak) > 0:
            peaks.append(int((vvec_peak[peak_start] + vvec_peak[-1]) / 2))

        peaks.append(vnum)
        valleys = np.array(valleys)
        peaks = np.array(peaks)
        back = 0
        splits = []
        for front in peaks:
            choose = np.where((valleys >= back) & (valleys < front))[0]
            if len(choose) == 1:
                splits.append(valleys[choose[0]])
            if len(choose) > 1:
                middle = (front + back) / 2
                mindist = vnum
                bestvalley = -1
                for vl in valleys[choose]:
                    curdist = abs(vl - middle)
                    if curdist < mindist:
                        bestvalley = vl
                        mindist = curdist
                splits.append(bestvalley)
            back = front

        # n = 3
        # if n + 1 < len(splits):
        #     sortedval_splits = np.argsort(list((map(lambda x: vvec[x], splits))))[::1]
        #     sortedval_splits = np.where(sortedval_splits < n-1)[0]
        #     splits = list(map(lambda x: splits[x], sortedval_splits))
        if n + 1 < len(splits):
            midarr = splits[1:-1]
            sortedval_splits = np.argsort(list((map(lambda x: vvec[x], midarr))))[::-1]
            sortedval_splits = np.where(sortedval_splits < n - 1)[0]
            midarr = list(map(lambda x: midarr[x], sortedval_splits))
            splits = [splits[0]] + midarr + [splits[-1]]

        splitted_im = []
        regsize = 28

        for i in range(len(splits) - 1):
            if splits[i + 1] - splits[i] < hnum / 10:
                continue
            curimg = im.crop((splits[i], 0, splits[i + 1], hnum))
            # print(splits)
            curimg = curimg.resize(
                (int(curimg.size[0] * regsize / curimg.size[1]), regsize),
                resample=Image.ANTIALIAS)
            arrayed = np.array(curimg)  # [:, :, 0]
            if arrayed.ndim > 2:
                arrayed = arrayed[:, :, 0]

            if np.max(arrayed) > 0:
                factor = 255 / np.max(arrayed)
                arrayed = 255 - arrayed * factor

            # 去除像素量小于minconn_eliminate的连通分量
            minconn_eliminate = 50
            connect_limit = 100
            w_curimg = arrayed.shape[1]
            h_curimg = arrayed.shape[0]
            for cx in range(w_curimg):  # i-w-x j-h-y
                for cy in range(h_curimg):
                    if arrayed[cy][cx] >= connect_limit:
                        cur_conn = [[cx, cy]]
                        # print(i, j)
                        while True:
                            flood = False  # whether find any additonal flood
                            for posi, posj in cur_conn:
                                # left point
                                if (posi - 1 > 0) and (not [posi - 1, posj] in cur_conn) \
                                        and (arrayed[posj][posi - 1] > connect_limit):
                                    cur_conn.append([posi - 1, posj])
                                    flood = True
                                # right point
                                if (posi + 1 < w_curimg) and (not [posi + 1, posj] in cur_conn) \
                                        and (arrayed[posj][posi + 1] > connect_limit):
                                    cur_conn.append([posi + 1, posj])
                                    flood = True
                                # up point
                                if (posj - 1 > 0) and (not [posi, posj - 1] in cur_conn) \
                                        and (arrayed[posj - 1][posi] > connect_limit):
                                    cur_conn.append([posi, posj - 1])
                                    flood = True
                                # down point
                                if (posj + 1 < h_curimg) and (not [posi, posj + 1] in cur_conn) \
                                        and (arrayed[posj + 1][posi] > connect_limit):
                                    cur_conn.append([posi, posj + 1])
                                    flood = True
                            if not flood:
                                break
                        if len(cur_conn) < minconn_eliminate:
                            for posi, posj in cur_conn:
                                arrayed[posj][posi] = 0

            curimg = Image.fromarray(arrayed)

            # 求重心
            curw = arrayed.shape[1]
            sumxw = 0
            sumyw = 0
            sumw = 0
            for iii in range(regsize):
                for jjj in range(curw):
                    sumxw += jjj * arrayed[iii][jjj]
                    sumyw += iii * arrayed[iii][jjj]
                    sumw += arrayed[iii][jjj]
            if sumw == 0:
                xcenter = 0
                ycenter = 0
            else:
                xcenter = sumxw / sumw
                ycenter = sumyw / sumw
            # print(xcenter, ycenter)
            # leftpadding = int((regsize - curw) / 2)
            # rightpadding = regsize - curw - leftpadding
            padcurimg = curimg.crop((xcenter - regsize / 2,
                                     ycenter - regsize / 2,
                                     xcenter + regsize / 2,
                                     ycenter + regsize / 2))

            splitted_im.append(np.array(padcurimg))

        netimages = []
        for img in splitted_im:
            netimages.append(img.flatten())
        return np.array(netimages)
