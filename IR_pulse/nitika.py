import cv2

# 画像の読み込み
img = cv2.imread("/Volumes/Extreme SSD/masaya_eye/gantei211/2023-07-22 11-51-41.375_gantei211_1_0.tiff", 0)

# 閾値の設定
threshold = 140
ret2, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

#閾値がいくつになったか確認
print("ret2: {}".format(ret2))
# # 二値化(閾値100を超えた画素を255にする。)
# ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

# 二値化画像の表示
cv2.imshow("img_th", img_otsu)
cv2.waitKey()
cv2.destroyAllWindows()