import cv2
import numpy as np
import glob
import sys
import csv
import re
import os

dir_name = '/Volumes/Extreme SSD/masaya_eye/gantei211/'
files = glob.glob(dir_name+'*')
# OUTPUT_DIR='/Users/masayakinefuchi/imageSensing/RGB_pulse/_skinColorSeparation/result/'
# subject ='gantei'
# OUTPUT_FILE =OUTPUT_DIR +subject +'.csv'
num = len(files)

##import picture
img_name = files[0]
#img_name ="/Users/masayakinefuchi/imageSensing/IR_pulse/g1.jpg"
img = cv2.imread(img_name)
#img_copy=cv2.imread(img_name)
#Sobel処理


#グレースケール化
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('image.jpg',img_gray)
#Sobel処理
dst = cv2.Laplacian(img_gray, -1, ksize=3)    
cv2.imshow('image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()