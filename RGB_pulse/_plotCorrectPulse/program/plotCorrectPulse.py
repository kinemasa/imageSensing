import math
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt


def load_pulse(sample_rate, pulse_filename, save_filename):
    pulse_data = pd.read_csv(pulse_filename)
    
    start_time =20
    end_time =50
    start = sample_rate * start_time
    end = start + sample_rate * end_time+1

    extracted_pulse_data = pulse_data.iloc[start: end, 1]

    extracted_pulse_data.to_csv(save_filename, header=False, index=False)


def plot_part(subject, save_filename,OUTPUT_DIR,sample_rate):
    extracted_pulse_csv = np.loadtxt(save_filename, delimiter=",")
    save_filename = OUTPUT_DIR + subject + "_extracted_pulse_data_part.png"

    plt.plot(extracted_pulse_csv[sample_rate:sample_rate *6])
    plt.xticks([0,sample_rate,sample_rate*2, sample_rate*3,sample_rate*4,sample_rate*5])
    #plt.yticks([-200,-100,0,100,200])
    plt.savefig(save_filename)
    plt.close()


def plot_15s(subject, save_filename, dir):
    extracted_pulse_csv = np.loadtxt(save_filename, delimiter=",")
    save_filename = dir + subject + "_extracted_pulse_data_15s.png"

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    ax.plot(extracted_pulse_csv[:3840], color="green")
    ax.grid(color="gray", linestyle="--")
    plt.xticks([0, 1280, 2560, 3840])
    plt.savefig(save_filename)
    plt.close()


def plot_full(subject, save_filename, dir):
    extracted_pulse_csv = np.loadtxt(save_filename, delimiter=",")
    save_filename = dir + subject + "_extracted_pulse_data_full.png"

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    ax.plot(extracted_pulse_csv, color="green")
    ax.grid(color="gray", linestyle="--")
    plt.xticks([0, 1280, 2560, 3840, 5120, 6400, 7680])
    plt.savefig(save_filename)
    plt.close()


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
    peak1_index = signal.argrelmax(pulse, order=int(sample_rate / 3.5))[0]
    peak2_index = signal.argrelmin(pulse, order=int(sample_rate / 3.5))[0]

    return peak1_index, peak2_index


def visualize_pulse(pulse1, peak1_index, peak2_index, save_filename):
    """
    時系列データ(脈波を想定)を可視化する．

    Parameters
    ---------------
    pulse1 : np.ndarray (1dim / [データ長])
        BP脈波
    peak1_index: np.ndarray (1dim / [データ長])
        上側ピーク
    peak2_index : np.ndarray (1dim / [データ長])
        下側ピーク

    Returns
    ---------------
    Nothing

    """

    # 脈波データを可視化
    # fig1_ax2 = fig1_ax1.twinx()
    # fig1_ax2.plot(x, pulse[:used_length], color=cm.winter(0.1),linewidth=0.3, alpha=0.3)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    x = np.linspace(0, pulse1.shape[0], pulse1.shape[0])

    ax.plot(x, pulse1)

    plt.grid()
    plt.xticks([0, 1280, 2560, 3840, 5120, 6400, 7680])
    ax.scatter(x[peak1_index], pulse1[peak1_index], marker='x', color='green', s=100)
    ax.scatter(x[peak2_index], pulse1[peak2_index], marker='x', color='green', s=100)
    plt.savefig(save_filename)
    plt.close()


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
        data[:count_nan_lft] = np.nanmean(data[count_nan_lft + 1:count_nan_lft + 5])
    if count_nan_rgt > 0:
        data[-count_nan_rgt:] = np.nanmean(data[-count_nan_rgt - 5:-count_nan_rgt - 1])

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

    # PSD可視化
    #    seaborn.set_style("darkgrid")
    #    used_length = frq.shape[0]
    #    fig = plt.figure(figsize=(12, 9))
    #    ax = fig.add_subplot(111)
    #    # ax.set_ylim(0, 0.0001)
    #    x = np.linspace(1, used_length, used_length)
    #    ax.plot(frq[1:], psd[1:], color=cm.winter(0.3),linewidth=1.5, alpha=0.3)
    #     # ax.axvspan(0.04, 0.15, color=cm.winter(0.6), alpha=0.1)
    #     # ax.axvspan(0.151, 0.4, color=cm.winter(1), alpha=0.1)
    #     # ax.set_xlim([0, 1])
    #     # ax.set_ylim([0, 1])
    #    plt.savefig('PSD.png')
    #     # plt.show()

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

    print(ibi)
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

    timedmn_feature = np.array([ibi_mean, ibi_std, hr_mean, hr_std, rmssd, nn50, pnn50])

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

    # 呼吸数の計算
    hf_freq = frq[hf_indx]
    br_p = psd[hf_indx]
    br_freq = hf_freq[np.argmax(br_p)]
    br = br_freq * 60

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


def main():
   
    INPUT_DIR ='/Volumes/Extreme SSD/pulsedata2/'
    OUTPUT_DIR ='/Users/masayakinefuchi/imageSensing/RGB_pulse/_plotCorrectPulse/result/'   
    
    subject = "yamasaki32-me-HR"
    sample_rate = 256
    pulse_filename = INPUT_DIR + subject + ".txt"
    save_filename = OUTPUT_DIR + subject + "_extracted_pulse_data.csv"

    load_pulse(sample_rate, pulse_filename, save_filename)
    plot_part(subject, save_filename, OUTPUT_DIR,sample_rate)
    plot_15s(subject, save_filename, OUTPUT_DIR)
    plot_full(subject, save_filename, OUTPUT_DIR)

    pulse = np.loadtxt(save_filename, delimiter=",")
    peak1_index, peak2_index = detect_pulse_peak(pulse, sample_rate)


    save_pulse_marked = OUTPUT_DIR + subject + "_extracted_pulse_marked.png"
    visualize_pulse(pulse, peak1_index, peak2_index, save_pulse_marked)

    # ibi1, pulse_rate1, frq1, psd1 = calculate_hrv2(peak1_index, peak2_index, sample_rate)
    # t_feature = ibi_to_timedomain_feature(ibi1)
    # f_feature = psd_to_frqdomain_feature(frq1, psd1)
    # np.savetxt(OUTPUT_DIR+subject+"time_feature.csv", t_feature, delimiter=",")
    # np.savetxt(OUTPUT_DIR+subject+"freq_feature.csv", f_feature, delimiter=",")


if __name__ == "__main__":
    main()
