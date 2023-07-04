## 色素成分分離画像
import cv2
import numpy as np
import glob
import sys
import csv
import re
import os
import funcSkinSeparation as ss


## input directory name
dir_name = '/Volumes/Extreme SSD/ayumu/kao64-1/'
files = glob.glob(dir_name+'*')
num = len(files)
img_num = files[0]


##import image
img = cv2.imread(img_num)

## output name
OUTPUT_DIR='/Users/masayakinefuchi/imageSensing/RGB_pulse/_skinColorSeparation/result/'
subject ='murai3-1'
OUTPUT_FILE =OUTPUT_DIR +subject +'.png'

##output image
output = ss.skinSeparation(img,"Melanin")
outPutImg =output.astype(np.uint16)
cv2.imwrite(OUTPUT_FILE,outPutImg)
