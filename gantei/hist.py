import cv2
import matplotlib.pyplot as plt # ヒストグラム表示用

import cv2

# def onMouse(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, y)




def get_histogram(img):
    '''ヒストグラムの取得'''
    if len(img.shape) == 3:
        # カラーのとき
        channels = 3
    else:
        # モノクロのとき
        channels = 1

    histogram = []
    for ch in range(channels):
        # チャンネル(B, G, R)ごとのヒストグラム
        hist_ch = cv2.calcHist([img],[ch],None,[256],[0,256])
        histogram.append(hist_ch[:,0])

    # チャンネルごとのヒストグラムを返す
    return histogram

def draw_histogram(hist):
    '''ヒストグラムをmatplotlibで表示'''
    # チャンネル数
    ch = len(hist)

    # # グラフの表示色
    if (ch == 1):
        colors = ["black"]
        label = ["Gray"]
    else:
        colors = ["blue", "green", "red"]
        label = ["B", "G", "R"]

    # # ヒストグラムをmatplotlibで表示
    x = range(256)
    for col in range(ch):
        y = hist[col]
        plt.plot(x, y, color = colors[col], label = label[col])

    # 凡例の表示
    plt.legend(loc=2)

    plt.show()

######################################################################


# 画像の読込
img = cv2.imread("C:\\Users\\kine0\\labo\\imageSensing\\gantei\\gantei-sanko.tiff",cv2.IMREAD_UNCHANGED)


cv2.imshow('sample', img)
# cv2.setMouseCallback('sample', onMouse)
print(img[197,497,0])
cv2.waitKey(0)

# # 画像の表示
cv2.imshow("Image", img)

# # ヒストグラムの取得
hist = get_histogram(img)

# # ヒストグラムの描画
draw_histogram(hist)