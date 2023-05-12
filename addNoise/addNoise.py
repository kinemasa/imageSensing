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
INPUT_IMAGE =path /"data"/"houseki.png"

img = cv2.imread(str(INPUT_IMAGE))
h,w =img.shape[:2]

#ノイズ画像生成.
noise_level = 150
noise = np.random.randint(0,noise_level,(h,w,3))
##black
img = img + noise

#output image
OUTPUT_DIR=str(path) + "/noiseMaker"
OUTPUT_IMAGE = OUTPUT_DIR+"/output_" +'noise'+".png"
##output image
cv2.imwrite(OUTPUT_IMAGE,img)
