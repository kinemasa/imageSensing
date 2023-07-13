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

dir_name = '/Volumes/Extreme SSD/tumura3/gm-1/'
files = glob.glob(dir_name+'*')
OUTPUT_DIR='/Users/masayakinefuchi/imageSensing/RGB_pulse/_skinColorSeparation/result/'
subject ='gantei'
OUTPUT_FILE =OUTPUT_DIR +subject +'.csv'
num = len(files)

##import picture
img_name = files[0]
img = cv2.imread(img_name)
img_copy=cv2.imread(img_name)

width = int(img.shape[1])
height = int(img.shape[0])

roi = getVideoROI(img)
print(roi)
##select position ROI_size
width=50
height=50
x1 = 924
y1 = 450

#Crop Image(ROI)
selectRoi_crop = img[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]
#Crop   Image(selected_ROI)
fixedRoi_crop = img[int(y1):int(y1+height),int(x1):int(x1+width)]

cv2.rectangle(img_copy,
              pt1 =(x1,y1),
              pt2 =(x1+width,y1+height),
              color =(0,255,0),
              thickness =1,
              lineType =cv2.LINE_4,
              shift =0
              )
##output  ROI image
cv2.imwrite("selectedRoi.png", selectRoi_crop)
cv2.imwrite("fixedRoi.png", fixedRoi_crop)
cv2.imwrite("output.png",img_copy)




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
    
    ##ROI
    pulsewave[i] = np.mean(ss.skinSeparation(img[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2]),:],"Hemoglobin"))
    
    #fixedSIze
    #pulsewave[i] = np.mean(ss.skinSeparation(img[int(roi[1]):int(roi[1]+height),int(roi[0]):int(roi[0]+width),:],"Hemoglobin"))
    
    #fixedSize , fixed SIze
    #pulsewave[i] = np.mean(ss.skinSeparation(img[int(y1):int(y1+height),int(x1):int(x1+width),:],"Hemoglobin"))
    i += 1

    sys.stdout.flush()
    sys.stdout.write('\rProcessing... (%d/%d)' %(i,num))

i = 0

with open(OUTPUT_FILE, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for val in pulsewave:
        writer.writerow([val])
        i=i+1
