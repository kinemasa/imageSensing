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

dir_name = '/Volumes/Extreme SSD/ayumu/kao64-1/'
files = glob.glob(dir_name+'*')
OUTPUT_DIR='/Users/masayakinefuchi/imageSensing/RGB_pulse/_skinColorSeparation/result/'
subject ='ayumu64'
OUTPUT_FILE =OUTPUT_DIR +subject +'.csv'
num = len(files)
img_name = files[0]
img = cv2.imread(img_name)
width = int(img.shape[1])
height = int(img.shape[0])

roi = getVideoROI(img)
print(roi)
left, top, width, height = map(int,roi)
width = 500
height = 90
##脈波情報と時間を初期化する
pulsewave = np.zeros(int(num))
time = np.zeros(int(num))

i = 0
for f in files:

    basename=os.path.basename(f)
    root, ext = os.path.splitext(basename)
    f_sp = re.split('[-_ ]',root)

    #print(f_sp)
    ##ファイルの名前から時間を取得（icexpressの出力に依存する）
    time[i] = float(f_sp[3])*3600000+float(f_sp[4])*60000+float(f_sp[5])*1000

    img = cv2.imread(f)
    ##平均画素値を色素成分関数に入力して得られたヘモグロビン画像の値で取得
    pulsewave[i] = np.mean(ss.skinSeparation(img[top:top + height, left: left +width ,:]))
    i += 1

    sys.stdout.flush()
    sys.stdout.write('\rProcessing... (%d/%d)' %(i,num))

i = 0

with open(OUTPUT_FILE, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for val in pulsewave:
        writer.writerow([val])
        i=i+1
