# 参考サイト：https://prog-you.com/mp4/

import glob
import cv2
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


# 入力動画
INPUT_VIDEO = '/Volumes/Extreme SSD/yamasaki3/yamasaki32-hr/*.bmp'
#INPUT_VIDEO = '/Users/user/Desktop/位置補正/output/img/0619/14-55-05/*.jpg'
# 出力動画 - ファイル名
OUTPUT_VIDEO = '/Volumes/Extreme SSD/yamasaki3/yamasaki32-hr/output.mp4'
# 出力動画 - フレームレート
OUTPUT_FPS = 60

img_array = []
for filename in sorted(glob.glob(INPUT_VIDEO), key=natural_keys):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

video = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), OUTPUT_FPS, size)

for i in range(len(img_array)):
    video.write(img_array[i])

video.release()