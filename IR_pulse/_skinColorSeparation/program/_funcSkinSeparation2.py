import numpy as np

def skinSeparation(GBR_img,output):
    """
    openCVのimreadで読み込んだnumpy形式の画像を入力する．
    出力はunsigned 16bit int (uint16) 形式のヘモグロビン画像であるが，
    出力する段階ではdoubleである．
    画像化する場合は出力された画像に対してnp.uint16(img)でキャストする必要がある．
    funcSkinSeparationは濃度空間から一度戻して画像化しているが，こちらは濃度をそのまま吐き出す．
    peak上側検出のプログラム用に反転して出力
    """
    #import imagefile
    height, width, channels = GBR_img.shape[:3]
    Img_size = height * width
    
     # color element
    melanin    = np.array([ 0.4143, 0.3570, 0.8372 ])
    hemoglobin = np.array([ 0.2988, 0.6838, 0.6657 ])
    shading    = np.array([ 1.0000, 1.0000, 1.0000 ])
    
    vec =np.vstack([shading,melanin,hemoglobin])
            
 
    
    # draw a normal line
   
    housen = [vec[1,1]*vec[2,2]-vec[1,2]*vec[2,1], vec[1,2]*vec[2,0]-vec[1,0]*vec[2,2], vec[1,0]*vec[2,1]-vec[1,1]*vec[2,0]]
    
    MinSkin = np.array([0, 0, 0])
    
    # BIas
    Bias = MinSkin
    
    ##　Making Mask
    Mask = np.copy(GBR_img[:,:,0])
    Mask = np.where(Mask > 0, 1, 0)
    
    # init 
    DC = 1/255.0
    L = np.zeros((height*width*channels,1))
    linearSkin = np.zeros((channels,Img_size))
    RGB＿log = np.zeros((channels,Img_size)) 

   ##BGR -> RGB

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
    
    ##skin element
    skin = np.array([img_r[:], img_g[:], img_b[:]])
    # -------------------------------------------------------------
    # gamma correction
    aa = 1
    bb = 0
    gamma = 1
    cc = 0
    gg = [1, 1, 1]
    for j in range(channels):
       linearSkin[j] = (((skin[j,:].astype(np.float64)-cc)/aa)*(1/gamma)-bb)/gg[j]/255

    # making mask
    img_mask  = np.ones((channels,Img_size))   # mask(0 or 1)
    img_mask2 = np.zeros((channels,Img_size))       # mask(DC or 0)
    
    img_mask = np.where(linearSkin == 0, 0, 1)
    img_mask2 = np.where(linearSkin == 0, DC, 0)

    # density space of color
    for j in range(channels):
       linearSkin[j] = linearSkin[j] + img_mask2[j]
       RGB_log[j] = -np.log(linearSkin[j])
    
    RGB_log = RGB_log * img_mask.astype(np.float64)
    
    for i in range(channels):
       RGB_log[i] = RGB_log[i] - MinSkin[i]

    # 照明ムラ方向と平行な成分をとおる直線と肌色分布平面との交点を求める
    # housen：肌色分布平面の法線
    # vec：独立成分ベクトル
    
    intersection = -(np.dot(housen[0],RGB_log[0])+np.dot(housen[1],RGB_log[1])+np.dot(housen[2],RGB_log[2]))/(np.dot(housen[0],vec[0,0])+np.dot(housen[1],vec[0,1])+np.dot(housen[2],vec[0,2]))
    
    # 陰影除去
    # skin_flat：陰影除去したメラヘモベクトルの平面
    # rest：陰影成分
    skin_flat = np.dot(intersection[:,np.newaxis],vec[0,:][np.newaxis,:]).T + RGB_log
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
    elif(output =="Melanin"):
       out = Mel
    else:
       out = Shadow
       
    return  -out
   