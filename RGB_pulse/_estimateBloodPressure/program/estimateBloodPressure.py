import glob
import os
import sys
import time
import math
import csv

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.sparse import spdiags
import matplotlib.pyplot as plt


""" Constant Setting """
SAMPLE_RATE = 60 # データのサンプルレート[fps]


""" Global Variables """
count_flag = 0

def interpolate_nan(data):
    """
    nanを近傍データで補間 (1次元 or 2次元)

    Parameters
    ---------------
    data : 1D [データ長] or 2D [データ数, データ長] / データ長軸で補間
        補間対象データ

    Returns
    ---------------
    data_inpl : 1D or 2D
        nanを補間したデータ

    """

    x = np.arange(data.shape[0])
    nan_indx = np.isnan(data)
    not_nan_indx = np.isfinite(data)

    # 始端，終端に1つもしくは連続してnanがあれば，左右近傍ではなく，片側近傍で補間
    count_nan_lft = 0
    count = 0
    while True:
        if not_nan_indx[count] == False:
            count_nan_lft += 1
            not_nan_indx[count] = True
        else:
            break
        count += 1

    count_nan_rgt = 0
    count = 1
    while True:
        if not_nan_indx[-count] == False:
            count_nan_rgt += 1
            not_nan_indx[-count] = True
        else:
            break
        count += 1

    if count_nan_lft > 0:
        data[:count_nan_lft] = np.nanmean(data[count_nan_lft+1:count_nan_lft+5])
    if count_nan_rgt > 0:
        data[-count_nan_rgt:] = np.nanmean(data[-count_nan_rgt-5:-count_nan_rgt-1])

    func_inpl = interp1d(x[not_nan_indx], data[not_nan_indx])
    data[nan_indx] = func_inpl(x[nan_indx])


    return data



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
    #error_indx = np.where((ibi < 0.33) | (1.5 < ibi))
    #flag[error_indx] = False
    #ibi[error_indx] = np.nan

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
        frq = np.fft.fftfreq(n=sn, d=1/fs)
        sn_hlf = math.ceil(sample_num / 2)
        psd = psd[:sn_hlf]
        frq = frq[:sn_hlf]

    else:
        ibi = np.nan
        pulse_rate = np.nan
        frq = np.nan
        psd = np.nan

    return ibi, pulse_rate, frq, psd


def ibi_to_timedomain_feature(ibi):
    """
    RR間隔から時間領域特徴量を抽出する．

    Parameters
    ---------------
    ibi : np.ndarray (1dim)
        IBIデータ

    Returns
    ---------------
    timedmn_feature : np.ndarray (1dim [特徴量数])
        時間領域の特徴量

    """

    ibi_length = ibi.shape[0]
    ibi_dff = np.diff(ibi)
    # ibi_dff = ibi[0:-1] - ibi[1:]

    ibi_mean = np.mean(ibi)
    ibi_std = np.std(ibi)

    rmssd = np.power(ibi_dff, 2)
    rmssd = np.mean(rmssd)
    rmssd = np.sqrt(rmssd)

    nn50 = np.count_nonzero(np.abs(ibi_dff) > 0.05)
    pnn50 = nn50 / ibi_length - 1

    nn50 = nn50 * 0.1

    hr = 60 / ibi
    # hr_mean = np.mean(hr)
    hr_mean = 60 / np.mean(ibi)
    hr_std = np.std(hr)

    timedmn_feature = np.array([ibi_mean, ibi_std, hr_mean, hr_std,
                                rmssd, nn50, pnn50])

    return timedmn_feature

def psd_to_frqdomain_feature(frq, psd):
    """
    IBIから周波数領域特徴量を抽出する．

    Parameters
    ---------------
    frq : np.ndarray (1dim)
        IBIのパワースペクトル密度の周波数軸
    psd : np.ndarray (1dim)
        IBIのパワースペクトル密度のパワー軸

    Returns
    ---------------
    frqdmn_feature : np.ndarray (1dim [特徴量数])
        周波数領域の特徴量

    """

    ulf_indx = np.where((frq < 0.0033))
    vlf_indx = np.where((0.0033 <= frq) & (frq < 0.04))
    lf_indx = np.where((0.04 <= frq) & (frq < 0.15))
    hf_indx = np.where((0.15 <= frq) & (frq < 0.4))

    #呼吸数の計算
    hf_freq = frq[hf_indx]
    br_p = psd[hf_indx]
    br_freq = hf_freq[np.argmax(br_p)]
    br = br_freq*60

    # lf_indx_shft = lf_indx[0] + 1
    # hf_indx_shft = hf_indx[0] + 1
    frq_dff = frq[1] - frq[0]

    # lf_frq_dff = frq[lf_indx_shft] - frq[lf_indx]
    # hf_frq_dff = frq[hf_indx_shft] - frq[hf_indx]

    ulf = np.sum(psd[ulf_indx] * frq_dff)
    vlf = np.sum(psd[vlf_indx] * frq_dff)
    lf = np.sum(psd[lf_indx] * frq_dff)
    hf = np.sum(psd[hf_indx] * frq_dff)

    # lf_peak = np.argmax(psd[lf_indx])
    # hf_peak = np.argmax(psd[hf_indx])
    # lf_peak = frq[lf_peak]
    # hf_peak = frq[hf_peak]

    # lf = np.sum(psd[lf_indx])
    # hf = np.sum(psd[hf_indx])

    n_lf = lf / (lf + hf)
    n_hf = hf / (lf + hf)

    ttl_indx = np.where((0.04 <= frq))
    ttl = np.sum(psd[ttl_indx])
    nn_lf = lf / ttl
    nn_hf = hf / ttl

    lfhf = lf / hf

    # frqdmn_feature = np.array([lf_peak, hf_peak, vlf, ulf, lf, hf, n_lf, n_hf, lfhf])
    frqdmn_feature = np.array([lf, hf, n_lf, n_hf, lfhf, br])

    return frqdmn_feature

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

    t1 = time.time()

    # デトレンドによりデータ終端は歪みが生じるため，1秒だけ切り捨てる．
    pulse_length = pulse.shape[0]
    used_length = pulse_length - sample_rate


    # デトレンド処理 / An Advanced Detrending Method With Application to HRV Analysis
    pulse_dt = np.zeros(pulse_length)
    order = len(str(pulse_length))
    lmd = sample_rate * 12 # サンプルレートによって調節するハイパーパラメータ
    wdth = sample_rate * 1 # データ終端が歪むため，データを分割してデトレンドする場合，wdth分だけ終端を余分に使用する．
    # if order > 4:
    if order > 4:
        splt = int(sample_rate / 40); # 40
        T = int(pulse_length/splt)
        # wdth = T
        for num in range(splt):
            print('\r\tDetrending Pulse %d / %d' %(num + 1, splt), end='')
            # sys.stderr.write('\r\tDetrending Pulse %d / %d' %(num, splt))
            # sys.stderr.flush()
            if num < (splt - 1):
                I = np.identity(T + wdth)
                flt = [1, -2, 1] * np.ones((1, T + wdth - 2), dtype=np.int64).T
                D2 = spdiags(flt.T, (range(0, 3)), T + wdth - 2, T + wdth)
                tmp = (I - np.linalg.inv(I + lmd**2 * D2.T @ D2)) @ pulse[num * T : (num + 1) * T + wdth]
                tmp = tmp[0, 0 : -wdth]
                pulse_dt[num * T : (num + 1) * T] = tmp
            else:
                I = np.identity(T)
                flt = [1, -2, 1] * np.ones((1, T-2), dtype=np.int64).T
                D2 = spdiags(flt.T, (range(0, 3)), T-2, T)
                pulse_dt[num * T : (num + 1) * T] = (I - np.linalg.inv(I + lmd**2 * D2.T @ D2)) @ pulse[num * T : (num + 1) * T]

    else:
        T = pulse_length
        I = np.identity(T)
        flt = [1, -2, 1] * np.ones((1, T-2), dtype=np.int64).T
        D2 = spdiags(flt.T, (range(0, 3)), T-2, T)
        pulse_dt[:] = (I - np.linalg.inv(I + lmd**2 * D2.T @ D2)) @ pulse

    pulse_dt = pulse_dt[0 : used_length]

    t2 = time.time()
    elapsed_time = int(t2 - t1)
    print(f'\tTime : {elapsed_time} sec')

    return pulse_dt



def bandpass_filter_pulse(pulse, band_width, sample_rate):
    """
    バンドパスフィルタリングにより脈波をデノイジングする．

    Parameters
    ---------------
    pulse : np.float (1 dim)
        脈波データ
    band_width : float (1dim / 2cmps)
        通過帯 [Hz] (e.g. [0.75, 4.0])
    sample_rate : int
        データのサンプルレート

    Returns
    ---------------
    pulse_sg : np.float (1 dim)
        デノイジングされた脈波

    """

    # バンドパスフィルタリング
    nyq = 0.5 * sample_rate
    b, a = signal.butter(1, [band_width[0]/nyq, band_width[1]/nyq], btype='band')
    pulse_bp = signal.filtfilt(b, a, pulse)

    return pulse_bp


def detect_pulse_peak(pulse, sample_rate):
    """
    脈波の上側ピークと下側ピークを検出する．

    Parameters
    ---------------
    pulse : np.float (1 dim)
        脈波データ
    sample_rate : int
        データのサンプルレート

    Returns
    ---------------
    peak1_indx : int (1 dim)
        脈波の上側ピーク
    peak2_indx : int (1 dim)
        脈波の下側ピーク

    """

    # ピーク検出
    peak1_indx = signal.argrelmax(pulse, order=int(sample_rate / 3.0))[0]
    peak2_indx = signal.argrelmin(pulse, order=int(sample_rate / 3.0))[0]


    return peak1_indx, peak2_indx



if __name__ == '__main__':
    
    INPUT_DIR= '/Users/masayakinefuchi/imageSensing/RGB_pulse/_skinColorSeparation/result/'
    subject = 'yamasaki2-open-10'
    INPUT_FILE = INPUT_DIR + subject + '.csv'
    OUTPUT_DIR='/Users/masayakinefuchi/imageSensing/RGB_pulse/_estimateBloodPressure/result/'
    OUTPUT_FILE = OUTPUT_DIR  + subject +'.csv'
    
    
    
    data = np.loadtxt(INPUT_FILE, delimiter=',')

#     # 分割あり，test
#     # 30[s]で計測
#     d_length = len(data)
#     roop_num = (d_length - SAMPLE_RATE * 15) / SAMPLE_RATE
#     d_start_point = 0
#     d_end_point = SAMPLE_RATE * 15
#     t_feature_array = np.zeros([7, int(roop_num)])
#     f_feature_array = np.zeros([6, int(roop_num)])

#     # 1Hzでサンプリング
#     for i in range(int(roop_num)):
# #        pulse = data[d_start_point:d_end_point]
#         pulse = data[d_start_point:d_end_point]
#         d_start_point += SAMPLE_RATE
#         d_end_point += SAMPLE_RATE

#         # 脈波信号のデトレンド処理
#         d_pulse = detrend_pulse(pulse, SAMPLE_RATE)

#         # d_pulseのバンドパスフィルタリング処理
#         bp_pulse = bandpass_filter_pulse(d_pulse, [0.75, 4.0], SAMPLE_RATE)

#         # ピーク検出
#         peak1, peak2 = detect_pulse_peak(bp_pulse, SAMPLE_RATE)

#         # 脈波データから心拍変動を算出
#         ibi1, pulse_rate1, frq1, psd1 = calculate_hrv2(peak1, peak2, SAMPLE_RATE)

#         t_feature = ibi_to_timedomain_feature(ibi1)
#         f_feature = psd_to_frqdomain_feature(frq1, psd1)

#         t_feature_array[:, i] = t_feature
#         f_feature_array[:, i] = f_feature
#         print(i, '/', roop_num)

#     np.savetxt('20220423_ex3_no2__gazou_bunkatu_time_feature.csv', t_feature_array, delimiter=',')
#     np.savetxt('20220423_ex3_no2_gazou_bunkatu_freq_feature.csv', f_feature_array, delimiter=',')

#     print('\r\n\nThis program has been finished successfully!')


    pulse = data

    # 脈波信号のデトレンド処理
    d_pulse = detrend_pulse(pulse, SAMPLE_RATE)

    # d_pulseのバンドパスフィルタリング処理
    bp_pulse = bandpass_filter_pulse(d_pulse, [0.75, 4.0], SAMPLE_RATE)

    # ピーク検出
    peak1, peak2 = detect_pulse_peak(bp_pulse, SAMPLE_RATE)

    # 脈波データから心拍変動を算出
    ibi1, pulse_rate1, frq1, psd1 = calculate_hrv2(peak1, peak2, SAMPLE_RATE)

    t_feature = ibi_to_timedomain_feature(ibi1)
    f_feature = psd_to_frqdomain_feature(frq1, psd1)

    # np.savetxt('time_feature.csv',t_feature,delimiter=',')
    # np.savetxt('freq_feature.csv',f_feature,delimiter=',')
    print(pulse_rate1)
