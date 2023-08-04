import cv2


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


img_1 = cv2.imread('sabuntest1.jpg',0)
img_2 = cv2.imread('sabuntest2.jpg',0)
img_diff = cv2.absdiff(img_1, img_2)
cv2.imwrite('sabun-result.jpg',img_diff)