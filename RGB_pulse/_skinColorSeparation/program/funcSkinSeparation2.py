# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:50:36 2021

@author: koike
"""
import numpy as np

def skinSeparation(img):
    """
    openCVのimreadで読み込んだnumpy形式の画像を入力する．
    出力はunsigned 16bit int (uint16) 形式のヘモグロビン画像であるが，
    出力する段階ではdoubleである．
    画像化する場合は出力された画像に対してnp.uint16(img)でキャストする必要がある．
    funcSkinSeparationは濃度空間から一度戻して画像化しているが，こちらは濃度をそのまま吐き出す．
    peak上側検出のプログラム用に反転して出力
    """
    melanin    = np.array([ 0.4143, 0.3570, 0.8372 ])
    hemoglobin = np.array([ 0.2988, 0.6838, 0.6657 ])
    shading    = np.array([ 1.0000, 1.0000, 1.0000 ])
            
    height, width, channels = img.shape[:3]
    Img_size = height * width
    Img_info = [height, width, 3, 1]
    
    # γ補正用パラメータの取得
    aa = 1
    bb = 0
    gamma = 1
    cc = 0
    gg = [1, 1, 1]
    
    # 色ベクトルと照明強度ベクトル
    vec = np.empty((3,3))
    vec[0,:] = shading
    vec[1,:] = melanin
    vec[2,:] = hemoglobin
    
    # 肌色分布平面の法線 = 2つの色ベクトルの外積
    # 平面から法線を求める式(vec(1,:) = [1 1 1]なので式には考慮せず)
    housen = [vec[1,1]*vec[2,2]-vec[1,2]*vec[2,1], vec[1,2]*vec[2,0]-vec[1,0]*vec[2,2], vec[1,0]*vec[2,1]-vec[1,1]*vec[2,0]]
    
    MinSkin = np.array([0, 0, 0])
    
    # バイアスベクトルの設定
    Bias = MinSkin

    Img_r = np.copy(img[:,:,2])
    Img_g = np.copy(img[:,:,1])
    Img_b = np.copy(img[:,:,0])

    Mask = np.copy(img[:,:,0])
    Mask = np.where(Mask > 0, 1, 0)

    temp_R = np.reshape(Img_r, Img_size)
    temp_G = np.reshape(Img_g, Img_size)
    temp_B = np.reshape(Img_b, Img_size)

    temp_RGB = np.array([temp_R, temp_G, temp_B]).T
    
    Original_Image = temp_RGB

    # 配列の初期化
    DC = 1/255.0;
    L = np.zeros((Img_info[0]*Img_info[1]*Img_info[2],1))
    linearSkin = np.zeros((Img_info[2],Img_size))
    S = np.zeros((Img_info[2],Img_size)) 

    img = Original_Image
    img = np.reshape(img, (Img_info[0], Img_info[1], Img_info[2]))

    img_r = np.reshape(img[:,:,0].T, height*width)
    img_g = np.reshape(img[:,:,1].T, height*width)
    img_b = np.reshape(img[:,:,2].T, height*width)

    skin = np.array([img_r[:], img_g[:], img_b[:]])
    # -------------------------------------------------------------

    # 画像のガンマ補正(画像の最大値を1に正規化)
    for j in range(Img_info[2]):
       linearSkin[j] = (((skin[j,:].astype(np.float64)-cc)/aa)*(1/gamma)-bb)/gg[j]/255

    # マスク画像の作成(黒い部分を除く)
    img_mask  = np.ones((Img_info[2],Img_size))   # マスク(0 or 1)
    img_mask2 = np.zeros((Img_info[2],Img_size))       # マスク(DC or 0)
    
    img_mask = np.where(linearSkin == 0, 0, 1)
    img_mask2 = np.where(linearSkin == 0, DC, 0)

    # 濃度空間(log空間)へ
    for j in range(Img_info[2]):
       linearSkin[j] = linearSkin[j] + img_mask2[j]
       S[j] = -np.log(linearSkin[j])
    
    S = S * img_mask.astype(np.float64)

    #ここまでメラヘモ以外の前処理
    # 肌色空間の起点を0へ
    for i in range(Img_info[2]):
       S[i] = S[i] - MinSkin[i]

    # 照明ムラ方向と平行な成分をとおる直線と肌色分布平面との交点を求める
    # housen：肌色分布平面の法線
    # S：濃度空間でのRGB
    # vec：独立成分ベクトル
    t = -(np.dot(housen[0],S[0])+np.dot(housen[1],S[1])+np.dot(housen[2],S[2]))/(np.dot(housen[0],vec[0,0])+np.dot(housen[1],vec[0,1])+np.dot(housen[2],vec[0,2]))
    
    # 陰影除去
    # skin_flat：陰影除去したメラヘモベクトルの平面
    # rest：陰影成分
    skin_flat = np.dot(t[:,np.newaxis],vec[0,:][np.newaxis,:]).T + S
    rest = S - skin_flat

    # *************************************************************
    # 色素濃度の計算
    # -------------------------------------------------------------
    # 混合行列と分離行列
    CompSynM = np.array([melanin, hemoglobin]).T
    CompExtM = np.linalg.pinv(CompSynM)
    # 各画素の色素濃度の取得
    #　　濃度分布 ＝ [メラニン色素；ヘモグロビン色素]
    #　　　　　　 ＝ 肌色ベクトル(陰影除去後)×分離行列
    Compornent = np.dot(CompExtM, skin_flat)

    # ヘモグロビン成分の補正(負数になってしまうため)
    # Compornent(2,:) = Compornent(2,:) + 0;
    # -------------------------------------------------------------
    Comp = np.vstack((Compornent, rest[0,:][np.newaxis,:]))
    
    temp_mhs = np.hstack([Comp[0,:], Comp[1,:], Comp[2,:]])[:,np.newaxis]
    L[:] = temp_mhs 
    
    #L_Mel = L[0:Img_size]
    L_Hem = L[Img_size:Img_size*2]
    
    return -L_Hem
   