import cv2
import numpy as np
from scipy.optimize import fmin
<<<<<<< HEAD
=======


>>>>>>> 9d1885efeac346d35b9ab2cee90a84ed3dea03d0

# 画像の読み込み
NShadow = '/Users/masayakinefuchi/imageSensing/RGB_pulse/_colorVectorEstimation/skin7.png'
SkinImage ='/Users/masayakinefuchi/imageSensing/RGB_pulse/_colorVectorEstimation/patch7.png'

# # 陰なし領域画像情報の取得
Nrgb = cv2.imread(NShadow).astype(np.float64)
Nr, Ng, Nb = Nrgb[:, :, 2], Nrgb[:, :, 1], Nrgb[:, :, 0]
Nheight, Nwidth, Nchannels = Nrgb.shape[:3]
NRGB = np.reshape(Nrgb,(Nheight*Nwidth,Nchannels))

# # 小領域画像情報の取得
rgb = cv2.imread(SkinImage).astype(np.float64)
r, g, b = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
height, width, channels = rgb.shape[:3]
RGB = np.reshape (rgb, (height*width ,channels))

# # 画像空間から濃度空間へ変換
Nrl, Ngl, Nbl = -np.log(Nr / 255), -np.log(Ng / 255), -np.log(Nb / 255)
rl, gl, bl = -np.log(r / 255), -np.log(g / 255), -np.log(b / 255)


# """
# 影のない領域でのメラニン　ヘモグロビンの値を算出する
# """

# # 肌色ベクトル
Nskin_vec = np.vstack((Nrl.flatten(), Ngl.flatten(), Nbl.flatten()))

# # # 肌色分布平面上のデータを主成分分析により白色化する
MeanNSKin_vec = np.mean(Nskin_vec, axis=1).reshape(-1, 1) ##平均
Nskin_Mat = np.kron(MeanNSKin_vec, np.ones((1, Nheight*Nwidth))) ##クロネッカー
Nskin_vecT =  Nskin_vec.T
NCov = np.cov(Nskin_vecT,rowvar=False)##分散共分散

_, NEigenvector = np.linalg.eig(NCov)##固有ベクトル

NEigenvector_1 = NEigenvector[:,-2:]
NEigenvector_2 =NEigenvector[:,0:1]

NEigenvector_1T =NEigenvector_1.T
NEigenvector_2T =NEigenvector_2.T

Nmelanin = NEigenvector_1[0:3,0]
Nhemoglobin = NEigenvector_1[0:3, 1]

# """
# 独立成分分析から色素成分ベクトルの推定（赤みのある部分）
# """
# # # # 肌色ベクトル
Skin_vec = np.vstack((rl.flatten(), gl.flatten(), bl.flatten()))

# # # # 陰影の除去
vec = np.vstack(([1, 1, 1], Nmelanin, Nhemoglobin))

# # 肌色分布平面の法線 = 2つの色ベクトルの外積
housen = [vec[1,1]*vec[2,2]-vec[1,2]*vec[2,1], vec[1,2]*vec[2,0]-vec[1,0]*vec[2,2], vec[1,0]*vec[2,1]-vec[1,1]*vec[2,0]]
# # 照明ムラ方向と平行な成分をとおる直線と肌色分布平面との交点を求める
t = -(np.dot(housen[0],Skin_vec[0])+np.dot(housen[1],Skin_vec[1])+np.dot(housen[2],Skin_vec[2]))/(np.dot(housen[0],vec[0,0])+np.dot(housen[1],vec[0,1])+np.dot(housen[2],vec[0,2]))

# # # 陰影除去
# # skin_flat 陰影除去したメラヘモベクトル
# # rest 陰影成分
skin_flat = np.dot(t[:,np.newaxis],vec[0,:][np.newaxis,:]).T + Skin_vec 
rest = Skin_vec - skin_flat

# # # # 肌色分布平面上のデータを，主成分分析により白色化する．
# # # # -------------------------------------------------------------
# # # # ゼロ平均計算用
MeanSkinFlat_vec = np.mean(skin_flat,axis=1).reshape(-1,1)
skin_Mat = np.kron(MeanSkinFlat_vec, np.ones((1, height*width))) ##クロネッカー
# # # # 濃度ベクトルSの固有値と固有ベクトルを計算
skin_flatT = skin_flat.T
Cov= np.cov(skin_flatT,rowvar=False)
_, Eigenvector = np.linalg.eig(Cov)

# # # # 第1主成分，第2主成分
Eigenvector1= Eigenvector[:,-2:]
Eigenvector2= Eigenvector[:,0:1]

Eigenvector1T= Eigenvector1.T
Eigenvector2T= Eigenvector2.T

Pcomponent = Eigenvector1T @(skin_flat-skin_Mat)
PcomponentP = Eigenvector2T @ (skin_flat-skin_Mat)

# # # # # 独立成分分析
# # Pcomponentの形状を取得
size_dimention, num_samples = Pcomponent.shape

Pstd = np.sqrt(np.mean(Pcomponent ** 2, axis=1))
NM = np.diag(1 / Pstd)

sensor = NM @ Pcomponent

res = sensor

# # """
# # # # Burelの独立評価値を最小化
# # """
<<<<<<< HEAD
# def factT(x):
#     countfactT = 1
#     for ifactT in range(1, x+1):
#         countfactT *= ifactT
#     return countfactT
=======
def factT(x):
    countfactT = 1
    for ifactT in range(1, x+1):
        countfactT *= ifactT
    return countfactT
>>>>>>> 9d1885efeac346d35b9ab2cee90a84ed3dea03d0

# def GfunkT(ga1,ga2,gb1,gb2):
#     Sigma = 1
#     G = 1

#     if (ga1 + gb1) % 2 == 0:
#         k = (ga1 + gb1) // 2
#         J2k = (factT(2 * k) * (2 * np.pi) ** 0.5) / (((4 ** k) * factT(k)) * (Sigma ** (2 * k - 1)))
#         sg = ((-1) ** ((ga1 - gb1) // 2) * J2k) / (factT(ga1) * factT(gb1))
#         G *= sg
#     else:
#         G = 0

#     if (ga2 + gb2) % 2 == 0:
#         k = (ga2 + gb2) // 2
#         J2k = (factT(2 * k) * (2 * np.pi) ** 0.5) / (((4 ** k) * factT(k)) / (Sigma ** (2 * k - 1)))
#         sg = ((-1) ** ((ga2 - gb2) // 2) * J2k) / (factT(ga2) * factT(gb2))
#         G *= sg
#     else:
#         G = 0
#     y = G
#     return y

# def fmin_Cal_Cost_Burel(K,M,GfuncT):
#     CostGMM = 0
#     for a1 in range(K+1):
#         for a2 in range(K+1-a1):
#             for b1 in range(K+1):
#                 for b2 in range(K+1-b1):
#                     CostGMM += GfuncT(a1, a2, b1, b2) * M[a1, a2] * M[b1, b2]
#     return CostGMM

# def fmin_Make_M(K,res):
#     M = np.zeros((K+1, K+1))
#     for m1 in range(K+1):
#         for m2 in range(K+1-m1):
#             E12 = np.mean((res[0, :]**m1) * (res[1, :]**m2))
#             E1E2 = np.mean(res[0, :]**m1) * np.mean(res[1, :]**m2)
#             M[m1, m2] =E12 - E1E2 


#     return M

# def f_burel(s):
#     x1, y1, x2, y2 = np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1])
#     H = np.array([[x1, y1], [x2, y2]])
#     res = H @ sensor

<<<<<<< HEAD
#     # Cost Cal.
#     K = 3
#     M = fmin_Make_M(K,res)
=======
    # Cost Cal.
    K = 4
    M = fmin_Make_M(K,res)
>>>>>>> 9d1885efeac346d35b9ab2cee90a84ed3dea03d0

#     CostGMM = fmin_Cal_Cost_Burel(K,M,GfunkT)
    
#     return CostGMM




<<<<<<< HEAD
# while True:
#     s = np.random.rand(2) * np.pi
#     # s = fmin(f_burel, s, xtol=1e-4, ftol=1e-8)
#     # f_burel(s)
#     x1, y1, x2, y2 = np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1])
#     H = np.array([[x1, y1], [x2, y2]])

#     TM = H @ NM @ PV
#     InvTM = np.linalg.pinv(TM)

#     c_1 = InvTM @ np.array([[1], [0]])
#     c_2 = InvTM @ np.array([[0], [1]])

#     for i in range(3):
#         if c_1[i, 0] < 0:
#             c_1[i, 0] *= -1
#         if c_2[i, 0] < 0:
#             c_2[i, 0] *= -1

#     if c_1[2, 0] > c_1[1, 0]:
#         melanin = c_1 / np.linalg.norm(c_1)
#         hemoglobin = c_2 / np.linalg.norm(c_2)
#     else:
#         melanin = c_2 / np.linalg.norm(c_2)
#         hemoglobin = c_1 / np.linalg.norm(c_1)
#     if (c_1 > 0).all() and (c_2 > 0).all():
#         break
=======
while True:
    s = np.random.rand(2) * np.pi
    s = fmin(f_burel, s, xtol=1e-4, ftol=1e-8)
    costGMM =f_burel(s)
    x1, y1, x2, y2 = np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1])
    H = np.array([[x1, y1], [x2, y2]])

    TM = H @ NM @ Eigenvector1T
 
    InvTM = np.linalg.pinv(TM)

    c_1 = InvTM @ np.array([[1], [0]])
    c_2 = InvTM @ np.array([[0], [1]])
 

    for i in range(3):
        if c_1[i, 0] < 0:
            c_1[i, 0] *= -1
        if c_2[i, 0] < 0:
            c_2[i, 0] *= -1
    
    print(c_1[2,0])
    print(c_1[1,0])
    if c_1[2, 0] > c_1[1, 0]:
        melanin = c_1 / np.sqrt(np.sum(pow(c_1,2)))
        hemoglobin = c_2 / np.sqrt(np.sum(pow(c_2,2)))
    else:
        melanin = c_2 / np.sqrt(np.sum(pow(c_2,2)))
        hemoglobin = c_1 / np.sqrt(np.sum(pow(c_1,2)))
        
    if (c_1 > 0).all() and (c_2 > 0).all():
        break
>>>>>>> 9d1885efeac346d35b9ab2cee90a84ed3dea03d0

#     print('エラー：色ベクトルが負の値です．')
#     flag = input('再試行：0 終了：1\n')
#     if flag == '1':
#         quit()

<<<<<<< HEAD
# # # # 色素濃度の最小値を求める
# CompSynM = np.vstack((melanin.T, hemoglobin.T, np.array([[1, 1, 1]])))
# CompExtM = np.linalg.pinv(CompSynM)
# Compornent = CompExtM @ skin_flat
# MinComp = np.min(Compornent[:3], axis=0)
# MinSkin = MinComp[0] * melanin + MinComp[1] * hemoglobin + MinComp[2] * np.array([1, 1, 1])

# # # # Excelファイルへ情報を書き込む
# filename = 'MelaHemo.csv'
# write = np.hstack((melanin.flatten(), hemoglobin.flatten(), s))
# np.savetxt(filename, write.reshape(1, -1), delimiter=',', fmt='%f')
# np.savetxt('melanin.csv', melanin.reshape(1, -1), delimiter=',', fmt='%f')
# np.savetxt('hemoglobin.csv', hemoglobin.reshape(1, -1), delimiter=',', fmt='%f')
=======
# # # # # # 色素濃度の最小値を求める
CompSynM = np.vstack((melanin.T, hemoglobin.T, np.array([[1, 1, 1]])))
CompExtM = np.linalg.pinv(CompSynM)
Compornent = CompExtM @ skin_flat
MinComp = np.min(Compornent[:3], axis=0)
MinSkin = MinComp[0] * melanin + MinComp[1] * hemoglobin + MinComp[2] * np.array([1, 1, 1])


# # # Excelファイルへ情報を書き込む
filename = 'MelaHemo.csv'
write = np.hstack((melanin.flatten(), hemoglobin.flatten(), costGMM))
np.savetxt(filename, write.reshape(1, -1), delimiter=',', fmt='%f')
np.savetxt('melanin.csv', melanin.reshape(1, -1), delimiter=',', fmt='%f')
np.savetxt('hemoglobin.csv', hemoglobin.reshape(1, -1), delimiter=',', fmt='%f')
>>>>>>> 9d1885efeac346d35b9ab2cee90a84ed3dea03d0
