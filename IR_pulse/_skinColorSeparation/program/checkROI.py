# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
import sys
import csv
import re
import os
import funcSkinSeparation2 as ss

def getVideoROI(img):
    roi = cv2.selectROI(img)
    cv2.destroyAllWindows()
    return roi

dir_name = '/Volumes/Extreme SSD/yamasaki3/yamasaki32-me-hr/'
files = glob.glob(dir_name+'*')
OUTPUT_DIR='/Users/masayakinefuchi/imageSensing/RGB_pulse/_skinColorSeparation/result/'
subject ='yamasaki32-me-upperLeft'
OUTPUT_FILE =OUTPUT_DIR +subject +'.csv'
num = len(files)
img_name = files[0]
img = cv2.imread(img_name)
width = int(img.shape[1])
height = int(img.shape[0])

roi = getVideoROI(img)
print(roi)
width=50
height=50

#Crop Image
selectRoi_crop = img[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]

fixedRoi_crop = img[int(roi[1]):int(roi[1]+height),int(roi[0]):int(roi[0]+width)]

cv2.imwrite("selectedRoi.png", selectRoi_crop)
cv2.imwrite("fixedRoi.png", fixedRoi_crop)