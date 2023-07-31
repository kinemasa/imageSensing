import time
import math
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from _interplolate_nan import interpolate_nan

def calculate_hrv2(peak1_indx, peak2_indx, sample_rate):
    """
    脈波データから心拍変動(IBI，心拍数，PSD)を算出する．

    Parameters
    ---------------
    pulse : np.ndarray (1dim)
        脈波データ
    sample_rate : int
        脈波のサンプルレート

    Returns
    ---------------
    ibi : nd.ndarray (1dim [100[fps] × 合計時間])
        IBI
    pulse_rate : np.float
        心拍数
    frq : np.ndarray (1dim)
        周波数軸
    psd : np.ndarray (1dim)
        パワースペクトル密度

    """

    # 下側ピーク数 / 心拍変動は下側ピークで算出した方が精度が良い．
    peak_num = peak2_indx.shape[0]
    ibi = np.zeros([peak_num - 1])
    flag = np.zeros([peak_num - 1])

    # IBI算出
    for num in range(peak_num - 1):
        ibi[num] = (peak2_indx[num + 1] - peak2_indx[num]) / sample_rate

    # ibiが[0.25, 1.5][sec]の範囲内に無い場合，エラーとする．/ [0.5, 1.5]
    # error_indx = np.where((ibi < 0.33) | (1.5 < ibi))
    # flag[error_indx] = False
    # ibi[error_indx] = np.nan

    global count_flag
    if np.any(flag):
        print('[!]')
        count_flag += 1

    ibi_num = ibi.shape[0]
    # スプライン補間は，次数以上の要素数が必要
    if ibi_num > 3:
        spln_kind = 'cubic'
    elif ibi_num > 2:
        spln_kind = 'quadratic'
    elif ibi_num > 1:
        spln_kind = 'slinear'
    else:
        ibi = np.nan

    total_time = np.sum(ibi)
    if np.isnan(total_time) != True:
        # エラーが発生した箇所の補間
        ibi = interpolate_nan(ibi)
        total_time = np.sum(ibi)
        # 心拍数の算出
        pulse_rate = 60 / np.mean(ibi)
        # スプライン補間 / 1fpsにリサンプリング
        # リサンプリングレート
        fs = 2
        sample_num = int(total_time * fs)
        x = np.linspace(0.0, total_time, ibi_num)
        f_spln = interp1d(x, ibi, kind=spln_kind)
        x_new = np.linspace(0.0, int(total_time), sample_num)
        ibi_spln = f_spln(x_new)

        sn = ibi_spln.shape[0]
        psd = np.fft.fft(ibi_spln)
        psd = np.abs(psd)
        frq = np.fft.fftfreq(n=sn, d=1 / fs)
        sn_hlf = math.ceil(sample_num / 2)
        psd = psd[:sn_hlf]
        frq = frq[:sn_hlf]

    else:
        ibi = np.nan
        pulse_rate = np.nan
        frq = np.nan
        psd = np.nan


    return ibi, pulse_rate, frq, psd