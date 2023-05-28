import numpy as np 
from numpy.lib.stride_tricks import as_strided
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.gridspec as gridspec
import seaborn as sns
import math
import cv2
from pathlib import Path
from skimage import data 
##define filter
def make_sharp_kernel(k: int):
  return np.array([
    [-1 / 9, -1 / 9, -1 / 9],
    [-1 / 9, k +8 / 9, -1 / 9],
    [-1 / 9, -1 / 9, -1 / 9]
  ], np.float32)
  
def make_laplacian():
  return np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
  ], np.float32)

def sobel_horizontal():
   return np.array([
    [-1/8, -2/8, -1/8],
    [0, 0, 0],
    [1/8, 2/8, 1/8]
  ], np.float32)



def sobel_vertical():
  return np.array([
    [-1/8, 0, 0],
    [-2/8, 0, 2/8],
    [-1/8, 0, 1/8]
  ], np.float32)

  
##import image
path = Path(__file__).parent
INPUT_IMAGE =path /"data"/"jack.png"
img = cv2.imread(str(INPUT_IMAGE))

##choose FIlter
kernel = make_laplacian()
img_En = cv2.filter2D(img,-1,kernel).astype("uint8")

##cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3)
##cv2.Laplacian(img,cv2.CV_32F)

##output
OUTPUT_DIR=str(path) +"/edge_emphasis"
OUTPUT_IMAGE = OUTPUT_DIR+"/output" +".png"


cv2.imwrite(OUTPUT_IMAGE,img_En)