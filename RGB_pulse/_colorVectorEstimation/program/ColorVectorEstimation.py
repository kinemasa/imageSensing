import cv2
import numpy as np
import _f_burel as f
from scipy.optimize import fmin

# 画像の読み込み
NShadow = 'C:\\Users\\kine0\\labo\\imageSensing\\RGB_pulse\\_colorVectorEstimation\skin7.png'
SkinImage = 'C:\\Users\\kine0\\labo\\imageSensing\\RGB_pulse\\_colorVectorEstimation\patch7.png'

# 陰なし領域画像情報の取得
Nrgb = cv2.imread(NShadow,cv2.COLOR_BGR2RGB).astype(np.float64)
Nr, Ng, Nb = Nrgb[:, :, 0], Nrgb[:, :, 1], Nrgb[:, :, 2]
Nheight, Nwidth, Nchannels = Nrgb.shape[:3]
# 小領域画像情報の取得
rgb = cv2.imread(SkinImage,cv2.COLOR_BGR2RGB).astype(np.float64)
r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
height, width, channels = rgb.shape[:3]

# 画像空間から濃度空間へ変換
Nrl, Ngl, Nbl = -np.log(Nr / 255), -np.log(Ng / 255), -np.log(Nb / 255)
rl, gl, bl = -np.log(r / 255), -np.log(g / 255), -np.log(b / 255)


"""
影のない領域でのメラニン　ヘモグロビンの値を算出する
"""

# 肌色ベクトル
Nskin_vec = np.vstack((Nrl.flatten(), Ngl.flatten(), Nbl.flatten()))

# 肌色分布平面上のデータを主成分分析により白色化する
MeanSKin_vec = np.mean(Nskin_vec, axis=1).reshape(-1, 1) ##平均
NSMMat = np.tile(MeanSKin_vec, (1, Nheight * Nwidth)) ##クロネッカー


NCov = np.cov(Nskin_vec)##分散共分散
_, Eigenvector = np.linalg.eig(NCov)##固有ベクトル

Eigenvector_twocol = Eigenvector[:, -2:]
Eigenvector_onecol =Eigenvector[:,:1]

Eigenvector_twocolT =Eigenvector_twocol.T
Eigenvector_onecolT =Eigenvector_onecol.T

Nmelanin = Eigenvector_twocol[:, 0]
Nhemoglobin = Eigenvector_twocol[:, 1]


"""
独立成分分析から色素成分ベクトルの推定
"""
# # 肌色ベクトル
Skin_vec = np.vstack((rl.flatten(), gl.flatten(), bl.flatten()))

# # 陰影の除去
vec = np.array([[1, 1, 1], Nmelanin, Nhemoglobin])

# # 肌色分布平面の法線 = 2つの色ベクトルの外積
housen = np.cross(vec[1], vec[2])

# # # 照明ムラ方向と平行な成分をとおる直線と肌色分布平面との交点を求める
numerator = -(housen[0] * Skin_vec[0, :] + housen[1] * Skin_vec[1, :] + housen[2] * Skin_vec[2, :])
denominator = housen[0] * vec[0, 0] + housen[1] * vec[0, 1] + housen[2] * vec[0, 2]
t = numerator / denominator
# # 陰影除去
removeShadow = (t[:, np.newaxis] * vec[0]).T + Skin_vec


# # 肌色分布平面上のデータを，主成分分析により白色化する．
# # -------------------------------------------------------------
# # ゼロ平均計算用
SM = np.mean(removeShadow, axis=1)
SMMat = np.tile(SM[:, np.newaxis], (1, height*width))

# # 濃度ベクトルSの固有値と固有ベクトルを計算
C = np.cov(removeShadow)
D, V = np.linalg.eig(C)

# # 第1主成分，第2主成分
PV = V[:, channels-2:channels].T
PVP = V[:, 0:1].T


# # 行列の積を計算

Pcomponent = PV @ (removeShadow - SMMat)
PcomponentP = PVP @ (removeShadow - SMMat)

# # # 独立成分分析
Pstd = np.sqrt(np.mean(Pcomponent ** 2, axis=1))
NM = np.diag(1 / Pstd)

sensor = NM @ Pcomponent
res = sensor

# # Burelの独立評価値を最小化
def f_burel(s):
    x1, y1, x2, y2 = np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1])
    H = np.array([[x1, y1], [x2, y2]])
    TM = H @ NM @ PV
    InvTM = np.linalg.pinv(TM)

#     # メラニン・ヘモグロビンの色素成分色ベクトル
    c_1 = InvTM @ np.array([[1], [0]])
    c_2 = InvTM @ np.array([[0], [1]])

    for i in range(3):
        if c_1[i, 0] < 0:
            c_1[i, 0] *= -1
        if c_2[i, 0] < 0:
            c_2[i, 0] *= -1

    if c_1[2, 0] > c_1[1, 0]:
        melanin = c_1 / np.linalg.norm(c_1)
        hemoglobin = c_2 / np.linalg.norm(c_2)
    else:
        melanin = c_2 / np.linalg.norm(c_2)
        hemoglobin = c_1 / np.linalg.norm(c_1)
    
#     # 評価値を逆数に変換して返す
    evaluation = 1.0 / np.linalg.norm(hemoglobin - melanin)
    return evaluation


while True:
    s = np.random.rand(2) * np.pi
    s = fmin(f_burel, s, xtol=1e-4, ftol=1e-8)

    x1, y1, x2, y2 = np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1])
    H = np.array([[x1, y1], [x2, y2]])

    TM = H @ NM @ PV
    InvTM = np.linalg.pinv(TM)

    c_1 = InvTM @ np.array([[1], [0]])
    c_2 = InvTM @ np.array([[0], [1]])

    for i in range(3):
        if c_1[i, 0] < 0:
            c_1[i, 0] *= -1
        if c_2[i, 0] < 0:
            c_2[i, 0] *= -1

    if c_1[2, 0] > c_1[1, 0]:
        melanin = c_1 / np.linalg.norm(c_1)
        hemoglobin = c_2 / np.linalg.norm(c_2)
    else:
        melanin = c_2 / np.linalg.norm(c_2)
        hemoglobin = c_1 / np.linalg.norm(c_1)

    if (c_1 > 0).all() and (c_2 > 0).all():
        break

    print('エラー：色ベクトルが負の値です．')
    flag = input('再試行：0 終了：1\n')
    if flag == '1':
        quit()

# # 色素濃度の最小値を求める
CompSynM = np.vstack((melanin.T, hemoglobin.T, np.array([[1, 1, 1]])))
CompExtM = np.linalg.pinv(CompSynM)
Compornent = CompExtM @ removeShadow
MinComp = np.min(Compornent[:3], axis=0)
MinSkin = MinComp[0] * melanin + MinComp[1] * hemoglobin + MinComp[2] * np.array([1, 1, 1])

# # Excelファイルへ情報を書き込む
filename = 'MelaHemo.csv'
write = np.hstack((melanin.flatten(), hemoglobin.flatten(), s))
np.savetxt(filename, write.reshape(1, -1), delimiter=',', fmt='%f')
np.savetxt('melanin.csv', melanin.reshape(1, -1), delimiter=',', fmt='%f')
np.savetxt('hemoglobin.csv', hemoglobin.reshape(1, -1), delimiter=',', fmt='%f')