# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:50:36 2021

@author: koike
"""
import numpy as np

def skinSeparation(img):
    """
    openCVのimreadで読み込んだnumpy形式の画像を入力する．
    配列として1pxごと以下のような形の配列として取り出される
    [[B G R],[B G R]]
    出力はunsigned 16bit int (uint16) 形式のヘモグロビン画像であるが，
    出力する段階ではdoubleである．
    画像化する場合は出力された画像に対してnp.uint16(img)でキャストする必要がある．
    funcSkinSeparationは濃度空間から一度戻して画像化しているが，こちらは濃度をそのまま吐き出す．
    peak上側検出のプログラム用に反転して出力
    """

    melanin    = np.array([ 0.4143, 0.3570, 0.8372 ])
    hemoglobin = np.array([ 0.2988, 0.6838, 0.6657 ])
    shading    = np.array([ 1.0000, 1.0000, 1.0000 ])
   
   ##高さ,幅,チャンネル数を画像から取得する
   
    height, width, channels = img.shape[:3]
    Img_size = height * width
    Img_info = [height, width, channels, 1]
    
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
    ##RGBのそれぞれの画像を取得
    Img_r = np.copy(img[:,:,2])
    Img_g = np.copy(img[:,:,1])
    Img_b = np.copy(img[:,:,0])
    
    ##画像のBの値を用いてマスクを生成する
    Mask = np.copy(img[:,:,0])
    Mask = np.where(Mask > 0, 1, 0)##条件,True,False
    ##配列を(1×n）から（width × height)へ
    temp_R = np.reshape(Img_r, Img_size)
    temp_G = np.reshape(Img_g, Img_size)
    temp_B = np.reshape(Img_b, Img_size)

    temp_RGB = np.array([temp_R, temp_G, temp_B]).T##転置
    ##元の画像をコピーした画像
    Original_Image = temp_RGB

    # 配列の初期化
    DC = 1/255.0 ##画素を０から１までの値にするための変数
    L = np.zeros((width*height*channels,1))##要素数が〜一列の配列
    linearSkin = np.zeros((channels,Img_size))##０行列の作成
    
    ##濃度空間の値の配列
    densitySpace = np.zeros((channels,Img_size)) 

    Previous_Image = Original_Image
    ##高さ、幅,channel数の多次元配列化
    Previous_Image= np.reshape( Previous_Image, (height, width, channels))
   ##(height,width)の形状からheight*widthの一列配列への変換
    img_r = np.reshape(img[:,:,0].T, height*width)
    img_g = np.reshape(img[:,:,1].T, height*width)
    img_b = np.reshape(img[:,:,2].T, height*width)

    skin = np.array([img_r[:], img_g[:], img_b[:]])
    # -------------------------------------------------------------

    # 画像のガンマ補正(画像の最大値を1に正規化)##其々の色に対してガンマ補正を行う
    for j in range(channels):
       linearSkin[j] = (((skin[j,:].astype(np.float64)-cc)/aa)*(1/gamma)-bb)/gg[j]/255

    # マスク画像の作成(黒い部分を除く)
    img_mask = np.zeros((channels,Img_size))      
    img_mask = np.where(linearSkin == 0, DC, 0)     # マスク(DC or 0)

    # 濃度空間(log空間)へ
    for j in range(channels):
       linearSkin[j] = linearSkin[j] + img_mask[j]
       densitySpace[j] = -np.log(linearSkin[j])
       
    ##濃度空間の変換後にマスク画像を適応
    densitySpace =  densitySpace* img_mask.astype(np.float64)

    ##ベクトルの開始位置を変更する
    # 肌色空間の起点を0へ
    for i in range(Img_info[2]):
        densitySpace[i] = densitySpace[i] - MinSkin[i]

    # 照明ムラ方向と平行な成分をとおる直線と肌色分布平面との交点を求める
    # housen：肌色分布平面の法線
    # densitySpace：濃度空間でのRGB
    # vec：独立成分ベクトル
    #t = -(法線との濃度空間RGB)の内積/法線と独立性成分のベクトルの内積
    t = -(np.dot(housen[0], densitySpace[0])+np.dot(housen[1], densitySpace[1])+np.dot(housen[2], densitySpace[2]))/(np.dot(housen[0],vec[0,0])+np.dot(housen[1],vec[0,1])+np.dot(housen[2],vec[0,2]))
    
    # 陰影除去
    # skin_flat：陰影除去したメラヘモベクトルの平面
    # shadowIncident：陰影成分
    skin_flat = np.dot(t[:,np.newaxis],vec[0,:][np.newaxis,:]).T +  densitySpace
    shadowIncident =  densitySpace - skin_flat

    # *************************************************************
    # 色素濃度の計算
    # -------------------------------------------------------------
    # 混合行列と分離行列
    CompSynM = np.array([melanin, hemoglobin]).T
    ##分離行列は混合行列の逆数
    CompExtM = np.linalg.pinv(CompSynM)
    # 各画素の色素濃度の取得
    #　　濃度分布 ＝ [メラニン色素；ヘモグロビン色素]
    #　　　　　　 ＝ 肌色ベクトル(陰影除去後)×分離行列
    
    ##Component メラニンとヘモグロビンのベクトル平面に分離行列をかける
    Compornent = np.dot(CompExtM, skin_flat)


    # -------------------------------------------------------------
    #メラニン、ヘモグロビンの成分の行列と他の箇所の配列を縦方向に結合する
    Comp = np.vstack((Compornent, shadowIncident[0,:][np.newaxis,:]))
    ##水平に結合して新しい配列（２、波長数*３,1)を作る
    temp_mhs = np.hstack([Comp[0,:], Comp[1,:], Comp[2,:]])[:,np.newaxis]
    L[:] = temp_mhs 
    
    #L_Mel = L[0:Img_size]
    L_Hem = L[Img_size:Img_size*2]
    
    return -L_Hem
   