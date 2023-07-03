# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
import sys
import csv
import re
import os
import funcSkinSeparation as ss


def getVideoROI(img):
    roi = cv2.selectROI(img)
    cv2.destroyAllWindows()
    return roi

dir_name = '/Volumes/Extreme SSD/ayumu2_open/mogi/'
files = glob.glob(dir_name+'*')

OUTPUT_DIR='/Users/masayakinefuchi/imageSensing/RGB_pulse/_skinColorSeparation/result/'
subject ='ayumu2-open'
OUTPUT_FILE =OUTPUT_DIR +subject +'.csv'
num = len(files)

img_num = files[10]
img = cv2.imread(img_num)
hem = ss.skinSeparation(img)
print(hem.dtype)
hemImg =hem.astype(np.uint16)
print(hemImg.dtype)
cv2.imwrite('output.png',hemImg)
# print(img)
# width = int(img.shape[1])
# height = int(img.shape[0])

# roi = getVideoROI(img)
# print(roi)
# left, top, width, height = map(int,roi)
# width = 500
# height = 90
# ##脈波情報と時間を初期化する
# pulsewave = np.zeros(int(num))
# time = np.zeros(int(num))

# i = 0
# for f in files: