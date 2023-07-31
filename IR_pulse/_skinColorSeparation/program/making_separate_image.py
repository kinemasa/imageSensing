## 色素成分分離画像
import cv2
import numpy as np
import glob
import sys
import csv
import re
import os
import _funcSkinSeparation as ss


## input directory name
dir_name = '/Volumes/Extreme SSD/muramatu/muramatu-kao-open/'
files = glob.glob(dir_name+'*')
num = len(files)
img_num = files[0]
#img_num = "/Volumes/Extreme SSD/murai3/murai32-me/2023-07-06 16-41-10.554_murai32-me_594_593.bmp"

##import image
img = cv2.imread(img_num)

## output name
OUTPUT_DIR='/Users/masayakinefuchi/imageSensing/RGB_pulse/_skinColorSeparation/result/'
subject ='muramatu'
OUTPUT_FILE =OUTPUT_DIR +subject +'.png'

##output image
output = ss.skinSeparation(img,"Hemoglobin")
outPutImg =output.astype(np.uint16)
cv2.imwrite(OUTPUT_FILE,outPutImg)
