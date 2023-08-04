#ライブラリインポート
import cv2
import numpy as np

#ベース画像の読み込み
pic=cv2.imread('C:\\Users\\kine0\\labo\\imageSensing\\gantei\\gantei611.tiff', cv2.IMREAD_GRAYSCALE)
#pic=cv2.imread("D:\\masaya_eye\\gantei611\\2023-07-31 12-53-45.659_gantei611_2_1.tiff", cv2.IMREAD_GRAYSCALE)
#疑似カラー化_JET
pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_JET)
cv2.imwrite('pseudo_color_jet.jpg',np.array(pseudo_color))
#疑似カラー化_HOT
pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_HOT)
cv2.imwrite('pseudo_color_hot.jpg',np.array(pseudo_color))
#疑似カラー化_HSV
pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_HSV)
cv2.imwrite('pseudo_color_hsv.jpg',np.array(pseudo_color))
#疑似カラー化_RAINBOW
pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_RAINBOW)
cv2.imwrite('pseudo_color_rainbow.jpg',np.array(pseudo_color))