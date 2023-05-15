import time
import math
import numpy as np
from scipy.sparse import eye, spdiags



def detrend_pulse(pulse, sample_rate):
    """
    脈波をデトレンドする．
    脈波が短すぎるとエラーを出力．(T < wdth の場合)

    Parameters
    ---------------
    pulse : np.float (1 dim)
        脈波データ
    sample_rate : int
        データのサンプルレート

    Returns
    ---------------
    pulse_dt : np.float (1 dim)
        デトレンドされた脈波

    """
    ##逆関数を求める
    def inv_jit(A):
        return np.linalg.inv(A)

    t1 = time.time()

    # デトレンドによりデータ終端は歪みが生じるため，1秒だけ切り捨てる．
    pulse_length = len(pulse)
    used_length = pulse_length - sample_rate

    # デトレンド処理 / An Advanced Detrending Method With Application to HRV Analysis
    pulse_dt = np.zeros(pulse_length) ##初期化
    order = len(str(pulse_length))##長さ
    lmd = sample_rate * 12  # サンプルレートによって調節するハイパーパラメータ
    wdth = sample_rate * 1  # データ終端が歪むため，データを分割してデトレンドする場合，wdth分だけ終端を余分に使用する．
    
    if order > 4:
        split = int(sample_rate / 10)  # 6
        T = int(pulse_length / split)#脈波を6分割
        # wdth = T
        for num in range(split):
            print('\r\t[Detrending Pulse] : %d / %d' % (num + 1, split), end='')
            if num < (split - 1):
                I = np.identity(T + wdth)##単位行列の設定
                diagonal_Component = np.ones([T + wdth - 2, 1]) * np.array([1, -2, 1])##[[1,2,1],[1,2,1]...]
                diagonal_Position = np.array([0,1,2])
                ##二次差分行列を作る
                D2 = spdiags(diagonal_Component.T, diagonal_Position, T + wdth - 2, T + wdth)
                
                ##x  = I -(I+λ^2D2(転置 )＊D2)
                tmp= I + lmd ** 2 * np.conjugate(D2.T) * D2
                
                ##逆関数
                inv_tmp = inv_jit(tmp)
                
                z_stat = (I - inv_tmp) @ pulse[num * T: (num + 1) * T + wdth]
               
                z_stat = z_stat[0: -wdth]
                pulse_dt[num * T: (num + 1) * T] = z_stat
            else:##最後の分割数のとき
                I = eye(T, T)
                I = np.identity(T)
                diagonal_Component = np.ones([T + wdth - 2, 1]) * np.array([1, -2, 1])
                ##2次差分行列を生成
                D2 = spdiags(diagonal_Component.T, diagonal_Position, T - 2, T)
                
                tmp = I + lmd ** 2 * np.conjugate(D2.T) * D2
                ##逆関数
                inv_tmp = inv_jit(tmp)
                z_stat = (I - inv_tmp) @ pulse[num * T: (num + 1) * T]
                pulse_dt[num * T: (num + 1) * T] = z_stat

    else:
        T = pulse_length
        # I = eye(T, T)
        I = np.identity(T)
        diagonal_Component = np.ones([T + wdth - 2, 1]) * np.array([1, -2, 1])
        diagonal_Position = np.array([0,1,2])
        D2 = spdiags(diagonal_Component.T, diagonal_Position, T - 2, T)
        
        tmp = I + lmd ** 2 * np.conjugate(D2.T) * D2
        inv_tmp = inv_jit(tmp)
        pulse_dt[:] = (I - inv_tmp) @ pulse
        

 
    pulse_dt = pulse_dt[0: used_length]


    t2 = time.time()
    elapsed_time = int((t2 - t1) * 10)
    print(f'\tTime : {elapsed_time * 0.1} sec')

    return pulse_dt

