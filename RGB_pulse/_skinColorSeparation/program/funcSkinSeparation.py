# -*- coding: utf-8 -*-

import numpy as np

def skinSeparation(GBR_img,output):
    """
    openCVのimreadで読み込んだnumpy形式の画像を入力する．
    出力はunsigned 16bit int (uint16) 形式のヘモグロビン画像であるが，
    出力する段階ではdoubleである．
    画像化する場合は出力された画像に対してnp.uint16(img)でキャストする必要がある．
    """
    #画像情報の取得
    height, width, channels = GBR_img.shape[:3]
    Img_size = height * width
    
     # 色ベクトルと照明強度ベクトル
    melanin    = np.array([ 0.4143, 0.3570, 0.8372 ])
    hemoglobin = np.array([ 0.2988, 0.6838, 0.6657 ])
    shading    = np.array([ 1.0000, 1.0000, 1.0000 ])
    
    vec =np.vstack([shading,melanin,hemoglobin])
            
    # γ補正用パラメータの取得
    aa = 1
    bb = 0
    gamma = 1
    cc = 0
    gg = [1, 1, 1]
    
    # 肌色分布平面の法線 = 2つの色ベクトルの外積
    # 平面から法線を求める式(vec(1,:) = [1 1 1]なので式には考慮せず)
    housen = [vec[1,1]*vec[2,2]-vec[1,2]*vec[2,1], vec[1,2]*vec[2,0]-vec[1,0]*vec[2,2], vec[1,0]*vec[2,1]-vec[1,1]*vec[2,0]]
    
    MinSkin = np.array([0, 0, 0])
    
    # バイアスベクトルの設定
    Bias = MinSkin
    
    ##Mask画像の生成
    Mask = np.copy(GBR_img[:,:,0])
    Mask = np.where(Mask > 0, 1, 0)
    
    # 配列の初期化
    DC = 1/255.0
    L = np.zeros((height*width*channels,1))
    linearSkin = np.zeros((channels,Img_size))
    RGB＿log = np.zeros((channels,Img_size)) 

   ##BGR -> RGBの形に画像配列を変換する

    Img_r = np.copy(GBR_img[:,:,2])
    Img_g = np.copy(GBR_img[:,:,1])
    Img_b = np.copy(GBR_img[:,:,0])

    temp_R = np.reshape(Img_r, Img_size)
    temp_G = np.reshape(Img_g, Img_size)
    temp_B = np.reshape(Img_b, Img_size)
    
   
    temp_RGB = np.array([temp_R, temp_G, temp_B]).T

    RGB_img = temp_RGB
    RGB_img = np.reshape(RGB_img, (height, width, channels))

    img_r = np.reshape(RGB_img[:,:,0].T, height*width)
    img_g = np.reshape(RGB_img[:,:,1].T, height*width)
    img_b = np.reshape(RGB_img[:,:,2].T, height*width)
    
    ##肌色成分の成分
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

    # 濃度空間へ変換しマスクへと
    for j in range(channels):
       linearSkin[j] = linearSkin[j] + img_mask2[j]
       RGB_log[j] = -np.log(linearSkin[j])
    
    RGB_log = RGB_log * img_mask.astype(np.float64)
    # 肌色空間の起点を0へ
    for i in range(channels):
       RGB_log[i] = RGB_log[i] - MinSkin[i]

    # 照明ムラ方向と平行な成分をとおる直線と肌色分布平面との交点を求める
    # housen：肌色分布平面の法線
    # vec：独立成分ベクトル
    
    t = -(np.dot(housen[0],RGB_log[0])+np.dot(housen[1],RGB_log[1])+np.dot(housen[2],RGB_log[2]))/(np.dot(housen[0],vec[0,0])+np.dot(housen[1],vec[0,1])+np.dot(housen[2],vec[0,2]))
    
    # 陰影除去
    # skin_flat：陰影除去したメラヘモベクトルの平面
    # rest：陰影成分
    skin_flat = np.dot(t[:,np.newaxis],vec[0,:][np.newaxis,:]).T + RGB_log
    rest = RGB_log - skin_flat

    # *************************************************************
    # 色素濃度の計算
    # -------------------------------------------------------------
    # 混合行列
    CompSynM = np.array([melanin, hemoglobin]).T
    # 分離行列
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
    Mel =temp_mhs[0:Img_size]
    Hem = temp_mhs[Img_size:Img_size*2]
    Shadow = temp_mhs[Img_size*2:Img_size*3]
    
    if (output == "Hemoglobin"):
       out = Hem
       out_vec = hemoglobin
    elif(output =="Melanin"):
       out = Mel
       out_vec = melanin
    else:
       out = Shadow
       out_vec = shading
       
    # -------------------------------------------------------------------------

    # *************************************************************************
    # 色素成分分離画像の出力
    # -------------------------------------------------------------------------
    SP_r = np.dot(vec[:,1][np.newaxis,:],Comp) + Bias[0]
    SP_g = np.dot(vec[:,2][np.newaxis,:],Comp) + Bias[1]
    SP_b = np.dot(vec[:,0][np.newaxis,:],Comp) + Bias[2]

    SP = np.array([SP_r, SP_g, SP_b])

    # 画像空間へ変換
    r_img = np.exp(-SP[0])
    g_img = np.exp(-SP[1])
    b_img = np.exp(-SP[2])

    r_img[r_img>1.0] = 1.0
    r_img[r_img<0.0] = 0.0
    g_img[g_img>1.0] = 1.0
    g_img[g_img<0.0] = 0.0
    b_img[b_img>1.0] = 1.0
    b_img[b_img<0.0] = 0.0

    r_img = np.reshape(r_img, (width,height))
    g_img = np.reshape(g_img, (width, height))
    b_img = np.reshape(b_img, (width, height))

    # データ結合
    I_exp = np.empty((height,width,3))
    I_exp[:,:,0] = r_img.T
    I_exp[:,:,1] = g_img.T
    I_exp[:,:,2] = b_img.T
    f_img = I_exp
    
    # マスク画像の作成
    Mask3 = np.empty((height,width,3))
    Mask3[:,:,0] = Mask
    Mask3[:,:,1] = Mask
    Mask3[:,:,2] = Mask
    f_img = f_img * Mask3.astype(np.float32)

    # 色素画像の取得
    L_Vec = np.zeros((height*width*channels, 1))
    L_Obj = np.zeros((channels, Img_size))
    
    for i in range(np.size(out[0])):
       # 色ベクトルに各濃度を重み付ける
       for j in range(channels):
          L_Obj[j,:] = np.dot(out_vec[j], out[:,i])

    temp_rgb = np.hstack([L_Obj[0,:], L_Obj[1,:], L_Obj[2,:]])
   
    img2 = np.reshape(temp_rgb, (channels, width,height)).T
    
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

    img_exp = np.empty((height,width,3))
    img_exp[:,:,0] = img_eb
    img_exp[:,:,1] = img_eg
    img_exp[:,:,2] = img_er
    
    ef_img = img_exp * Mask3.astype(np.float32)
    ef_img = ef_img * 65535.0
    
    # doubleの形式で画像を出力(png等で出力する場合にはuint16でキャストする必要あり)
    return ef_img