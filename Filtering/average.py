import numpy as np 
from numpy.lib.stride_tricks import as_strided
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
from pathlib import Path

##parentfile
path = Path(__file__).parent
INPUT_IMAGE =path /"data"/"jack.png"


img = cv2.imread(str(INPUT_IMAGE))

##opencv filtering
filter_size = 70

img_blur = cv2.blur(src=img ,ksize=(filter_size,filter_size))

##making filter
# average_filter = np.array([[1,1,1],[1,1,1],[1,1,1]],np.float32)/9
# img_blur = cv2.filter2D(img,-1,average_filter)

##output image
OUTPUT_DIR=str(path) + "/average_data"
OUTPUT_IMAGE = OUTPUT_DIR+"/output_" +str(filter_size)+"*"+str(filter_size)+".png"
##output image
cv2.imwrite(OUTPUT_IMAGE,img_blur)
