# -*- coding: utf-8 -*-

import numpy as np

def skinSeparation(img):
    """
    openCVのimreadで読み込んだnumpy形式の画像を入力する．
    出力はunsigned 16bit int (uint16) 形式のヘモグロビン画像であるが，
    出力する段階ではdoubleである．
    画像化する場合は出力された画像に対してnp.uint16(img)でキャストする必要がある．
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
    vec =np.vstack([shading,melanin,hemoglobin])
    
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
    L = np.zeros((height*width*channels,1))
    linearSkin = np.zeros((channels,Img_size))
    S = np.zeros((channels,Img_size)) 

    img = Original_Image
    img = np.reshape(img, (height, width, channels))

    img_r = np.reshape(img[:,:,0].T, height*width)
    img_g = np.reshape(img[:,:,1].T, height*width)
    img_b = np.reshape(img[:,:,2].T, height*width)

    skin = np.array([img_r[:], img_g[:], img_b[:]])
    # -------------------------------------------------------------

    # 画像のガンマ補正(画像の最大値を1に正規化)
    for j in range(channels):
       linearSkin[j] = (((skin[j,:].astype(np.float64)-cc)/aa)*(1/gamma)-bb)/gg[j]/255

    # マスク画像の作成(黒い部分を除く)
    img_mask  = np.ones((channels,Img_size))   # マスク(0 or 1)
    img_mask2 = np.zeros((channels,Img_size))       # マスク(DC or 0)
    
    img_mask = np.where(linearSkin == 0, 0, 1)
    img_mask2 = np.where(linearSkin == 0, DC, 0)

    # 濃度空間(log空間)へ
    for j in range(channels):
       linearSkin[j] = linearSkin[j] + img_mask2[j]
       S[j] = -np.log(linearSkin[j])
    
    S = S * img_mask.astype(np.float64)

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

    L_Hem = L[Img_size:Img_size*2]
    # -------------------------------------------------------------------------

    # *************************************************************************
    # 色素成分分離画像の出力
    # -------------------------------------------------------------------------
    SP_r = np.dot(vec[:,1][np.newaxis,:],Comp) + Bias[0]
    SP_g = np.dot(vec[:,2][np.newaxis,:],Comp) + Bias[1]
    SP_b = np.dot(vec[:,0][np.newaxis,:],Comp) + Bias[2]

    SP = np.array([SP_r, SP_g, SP_b])

    # 画像空間へ変換
    rp = np.exp(-SP[0])
    gp = np.exp(-SP[1])
    bp = np.exp(-SP[2])

    rp[rp>1.0] = 1.0
    rp[rp<0.0] = 0.0
    gp[gp>1.0] = 1.0
    gp[gp<0.0] = 0.0
    bp[bp>1.0] = 1.0
    bp[bp<0.0] = 0.0

    rp = np.reshape(rp, (width,height))
    gp = np.reshape(gp, (width, height))
    bp = np.reshape(bp, (width, height))

    # データ結合
    I_exp = np.empty((height,width,3))
    I_exp[:,:,0] = rp.T
    I_exp[:,:,1] = gp.T
    I_exp[:,:,2] = bp.T

    f_img = I_exp
    # マスク画像の作成
    Mask3 = np.empty((height,width,3))
    Mask3[:,:,0] = Mask
    Mask3[:,:,1] = Mask
    Mask3[:,:,2] = Mask
    f_img = f_img * Mask3.astype(np.float32)

    SL_Com = L_Hem

    # 色素画像の取得
    L_Vec = np.zeros((height*width*channels, 1))
    L_Obj = np.zeros((channels, Img_size))
    
    for i in range(np.size(SL_Com[0])):
       # 色ベクトルに各濃度を重み付ける
       for j in range(Img_info[2]):
          L_Obj[j,:] = np.dot(hemoglobin[j], L_Hem[:,i])

    temp_rgb = np.hstack([L_Obj[0,:], L_Obj[1,:], L_Obj[2,:]])
    L_Vec = temp_rgb

    img2 = np.reshape(L_Vec, (channels, width,height)).T
    
    img_er = np.exp(-img2[:,:,0])
    img_eg = np.exp(-img2[:,:,1])
    img_eb = np.exp(-img2[:,:,2])

    # データを丸める
    img_er = np.where(img_er > 1.0, 1.0, img_er)
    img_er = np.where(img_er < 0.0, 0.0, img_er)
    img_eg = np.where(img_eg > 1.0, 1.0, img_eg)
    img_eg = np.where(img_eg < 0.0, 0.0, img_eg)
    img_eb = np.where(img_eb > 1.0, 1.0, img_eb)
    img_eb = np.where(img_eb < 0.0, 0.0, img_eb)

    img_exp = np.empty((Img_info[0],Img_info[1],3))
    img_exp[:,:,0] = img_eb
    img_exp[:,:,1] = img_eg
    img_exp[:,:,2] = img_er
    
    ef_img = img_exp * Mask3.astype(np.float32)
    ef_img = ef_img * 65535.0
    
    # doubleの形式で画像を出力(png等で出力する場合にはuint16でキャストする必要あり)
    return ef_img