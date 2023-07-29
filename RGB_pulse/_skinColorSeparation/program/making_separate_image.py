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
# dir_name = 'D:\\muramatu\\muramatu-kao-open\\'

# files = glob.glob(dir_name+'*')
# num = len(files)
# img_num = files[0]
img_num = "C:\\Users\\kine0\\labo\\imageSensing\\RGB_pulse\\_colorVectorEstimation\\image.bmp"

##import image
img = cv2.imread(img_num)

## output name
OUTPUT_DIR='c:/Users/kine0/labo/imageSensing/RGB_pulse/_skinColorSeparation/result/'
subject ='image'
OUTPUT_FILE =OUTPUT_DIR +subject +'.png'

##output image
output = ss.skinSeparation(img,"Hemoglobin")
outPutImg =output.astype(np.uint16)
cv2.imwrite(OUTPUT_FILE,outPutImg)
