#ライブラリインポート
import cv2
import numpy as np

#ベース画像の読み込み
pic=cv2.imread('C:\\Users\\kine0\\labo\\imageSensing\\gantei\\gantei-sanko.tiff')
#pic=cv2.imread("D:\\masaya_eye\\gantei611\\2023-07-31 12-53-45.659_gantei611_2_1.tiff", cv2.IMREAD_GRAYSCALE)
height, width, channels = pic.shape[:3]

pic_B= pic[:,:,0]
pic_G= pic[:,:,1]
pic_R= pic[:,:,2]

# pic_B[:,:] =0
# pic_G[:,:] =0
# pic_R[:,:] =0S

print(pic_B)
print(pic_G)
print(pic_R)
re_pic =np.zeros([height ,width ,channels])
re_pic[:,:,0]= pic_B
re_pic[:,:,1]= pic_G
re_pic[:,:,2]= pic_R

cv2.imwrite("gantei.png",re_pic)

# cv2.imwrite("ganteiB.png",pic_B)
# cv2.imwrite("ganteiG.png",pic_G)
# cv2.imwrite("ganteiR.png",pic_R)