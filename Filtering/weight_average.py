import numpy as np 
from numpy.lib.stride_tricks import as_strided
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.gridspec as gridspec
import seaborn as sns
import math
import cv2
from pathlib import Path

##import image
path = Path(__file__).parent
INPUT_IMAGE =path /"data"/"jack.png"

img = cv2.imread(str(INPUT_IMAGE))

##define filter
filter_size =3
weight_filter = np.array([[10,20,10],[20,40,20],[10,20,10]],np.float32)/160

img_blur = cv2.filter2D(img,-1,weight_filter)

##output
OUTPUT_DIR=path +"/weight_ave_data"
OUTPUT_IMAGE = OUTPUT_DIR+"/output_" +str(filter_size)+"*"+str(filter_size)+".png"


cv2.imwrite(OUTPUT_IMAGE,img_blur)