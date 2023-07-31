import cv2
import matplotlib.pyplot as plt
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
# 画像をグレースケールで読み込む
image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

# ヒストグラムを計算する
histogram = cv2.calcHist([image], [0], None, [256], [0,256])

# ヒストグラムをプロットする
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(histogram)
plt.xlim([0, 256])
plt.show()