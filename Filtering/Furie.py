import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from skimage import data
from skimage.color import rgb2gray


path = Path(__file__).parent
INPUT_IMAGE =path /"data"/"honeycomb.jpg"
original = plt.imread(INPUT_IMAGE)
grayscale = rgb2gray(original)

im_freq = np.fft.fft2(grayscale)
h, w = im_freq.shape
#     im_freq = np.roll(im_freq, h//2, 0)
#     im_freq = np.roll(im_freq, w//2, 1)
im_freq = np.fft.fftshift(im_freq)
furie = np.log10(np.abs(im_freq)) * 20
    
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(furie, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")

##output
OUTPUT_DIR=str(path) +"/edge_emphasis"
OUTPUT_IMAGE = OUTPUT_DIR+"/furie" +".png"
plt.savefig(OUTPUT_IMAGE,dpi=150)

fig.tight_layout()
plt.show()

