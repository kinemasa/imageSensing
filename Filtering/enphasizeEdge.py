import numpy as np 
from numpy.lib.stride_tricks import as_strided
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.gridspec as gridspec
import seaborn as sns
import math
import cv2
from pathlib import Path

##define filter
def make_sharp_kernel(k: int):
  return np.array([
    [-k / 9, -k / 9, -k / 9],
    [-k / 9, 1 + 8 * k / 9, k / 9],
    [-k / 9, -k / 9, -k / 9]
  ], np.float32)
  
##import image
path = Path(__file__).parent
INPUT_IMAGE =path /"data"/"jack2.png"
img = cv2.imread(str(INPUT_IMAGE))

kernel = make_sharp_kernel(5)
img_En = cv2.filter2D(img,-1,kernel).astype("uint8")
##output
OUTPUT_DIR=str(path) +"/edge_emphasis"
OUTPUT_IMAGE = OUTPUT_DIR+"/output" +".png"


cv2.imwrite(OUTPUT_IMAGE,img)