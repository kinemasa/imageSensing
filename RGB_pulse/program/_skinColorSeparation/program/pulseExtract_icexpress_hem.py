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

INPUT_DIR ='/Users/masayakinefuchi/脈波推定/exp-sub/murai/Cam 1/'
OUTPUT_DIR ='/Users/masayakinefuchi/脈波推定/imageSensing/RGB_pulse/program/_skinColorSeparation/result/'
OUTPUT_FILE = OUTPUT_DIR +'murai1' +'.csv'



dir_name =INPUT_DIR
files = glob.glob(dir_name+'*')

num = len(files)

img_name = files[0]
img = cv2.imread(img_name)
width = int(img.shape[1])
height = int(img.shape[0])

selected_roi = getVideoROI(img)
print(selected_roi)

roi = [selected_roi[0],selected_roi[1],selected_roi[0]+100,selected_roi[1]+40]
print(roi)
pulsewave = np.zeros(int(num))
time = np.zeros(int(num))

i = 0
for f in files:

    basename=os.path.basename(f)
    root, ext = os.path.splitext(basename)
    f_sp = re.split('[-_ ]',root)

    #print(f_sp)

    time[i] = float(f_sp[3])*3600000+float(f_sp[4])*60000+float(f_sp[5])*1000

    img = cv2.imread(f)

    pulsewave[i] = np.mean(ss.skinSeparation(img[roi[1]: roi[1] + roi[3], roi[0]: roi[0] + roi[2],:]))
    i += 1

    sys.stdout.flush()
    sys.stdout.write('\rProcessing... (%d/%d)' %(i,num))

i = 0

with open(OUTPUT_FILE, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for val in pulsewave:
        writer.writerow([val])
        i=i+1
