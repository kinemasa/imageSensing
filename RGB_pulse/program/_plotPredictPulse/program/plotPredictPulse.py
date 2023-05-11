import time
import math
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.sparse import eye, spdiags
import matplotlib.pyplot as plt


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
    lmd = sample_rate * 12  # サンプルレートによって調節するハイパーパラメータ
    wdth = sample_rate * 1  # データ終端が歪むため，データを分割してデトレンドする場合，wdth分だけ終端を余分に使用する．
    
    if order > 4:
        split = int(sample_rate / 16)  # 3
        T = int(pulse_length / split)#脈波を3分割
        # wdth = T
        for num in range(split):
            print('\r\t[Detrending Pulse] : %d / %d' % (num + 1, split), end='')
            if num < (split - 1):
                I = np.identity(T + wdth)##指定した大きさ分の単位行列を作成する
                flt = np.ones([T + wdth - 2, 1]) * np.array([1, -2, 1])##
                ##2次差分行列を生成
                D2 = spdiags(flt.T, np.array([0, 1, 2]), T + wdth - 2, T + wdth)
                ##D2にD2の転置共役の積をかける=
                ##x  = I -(I+λ^2D2(転置 )＊D2)
                preinv = I + lmd ** 2 * np.conjugate(D2.T) * D2
                ##逆関数
                inv_tmp = inv_jit(preinv)
                
                tmp = (I - inv_tmp) @ pulse[num * T: (num + 1) * T + wdth]
               
                tmp = tmp[0: -wdth]
                pulse_dt[num * T: (num + 1) * T] = tmp
            else:##最後の分割数のとき（今回は３ばんめ）
                I = eye(T, T)
                I = np.identity(T)
                flt = np.ones([T + wdth - 2, 1]) * np.array([1, -2, 1])
                ##2次差分行列を生成
                D2 = spdiags(flt.T, np.array([0, 1, 2]), T - 2, T)
                
                preinv = I + lmd ** 2 * np.conjugate(D2.T) * D2
                ##逆関数
                inv_tmp = inv_jit(preinv)
                tmp = (I - inv_tmp) @ pulse[num * T: (num + 1) * T]
                pulse_dt[num * T: (num + 1) * T] = tmp

    else:
        T = pulse_length
        # I = eye(T, T)
        I = np.identity(T)
        flt = np.ones([T + wdth - 2, 1]) * np.array([1, -2, 1])
        D2 = spdiags(flt.T, np.array([0, 1, 2]), T - 2, T)
        preinv = I + lmd ** 2 * np.conjugate(D2.T) * D2
       
        inv_tmp = inv_jit(preinv)
        pulse_dt[:] = (I - inv_tmp) @ pulse
        

 
    pulse_dt = pulse_dt[0: used_length]


    t2 = time.time()
    elapsed_time = int((t2 - t1) * 10)
    print(f'\tTime : {elapsed_time * 0.1} sec')

    return pulse_dt


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
    peak1_index = signal.argrelmax(pulse, order=int(sample_rate / 3.0))[0]
    peak2_index = signal.argrelmin(pulse, order=int(sample_rate / 3.0))[0]

    return peak1_index, peak2_index


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
    b, a = signal.butter(1, [band_width[0] / nyq, band_width[1] / nyq], btype='band')
    pulse_bp = signal.filtfilt(b, a, pulse)

    return pulse_bp


def preprocess_pulse(pulse, sample_rate):
    """
    脈波に対して前処理を行う．
    振幅逆転 > デトレンド > SGフィルタ > バンドパスフィルタ > ピーク検出

    Parameters
    ---------------
    pulse_lst : list(1dim : np.ndarray (2dim, [フレーム数, 波長数]))
        ファイルごとの脈波
    sample_rate : int
        脈波のサンプルレート

    Returns
    ---------------
    pulse_dt : np.ndarray (2dim, [フレーム数, 波長数]))
        デトレンド脈波
    pulse_sg : np.ndarray (2dim, [フレーム数, 波長数]))
        SGフィルタリング脈波
    pulse_bp : np.ndarray (2dim, [フレーム数, 波長数]))
        BPフィルタリング脈波
    peak1_indx : np.ndarray (1dim, [ピーク数]))
        上側ピーク
    peak2_indx : np.ndarray (2dim, [ピーク数]))
        下側ピーク

    """

    # print('[Preprocessing Pulse]\n')

    # # 脈波に0が含まれている場合，近傍を用いて補間する．
    # pulse[pulse==0] = np.nan
    # pulse = btp.interpolate_nan(pulse)

    # # 脈波に異常値が含まれている場合，近傍を用いて補間する．
    # pulse, outlier_indx = btp.interpolate_outlier(pulse, True, th_constant=2.5)

    # デトレンド / 脈波の終端切り捨てが発生することに注意
    pulse_dt = detrend_pulse(pulse, sample_rate)

    # バンドパスフィルタリング / [0.75, 5.0]
    band_width = [0.75, 5.0]
    pulse_bp = bandpass_filter_pulse(pulse_dt, band_width, sample_rate)

    # ピーク検出
    peak1_index, peak2_index = detect_pulse_peak(pulse_bp, sample_rate)

    return pulse_dt, pulse_bp, peak1_index, peak2_index


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

    nn50 *= 0.1

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


def plot_part(input_filename, save_filename, wiener=False):
    data_input = np.loadtxt(input_filename, delimiter=",")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if wiener:
        ax.plot(data_input[:332], color="orangered")
    else:
        ax.plot(data_input[:332])
    ax.grid(color="gray", linestyle="--")
    plt.xticks([0, 66, 133, 199, 266, 332])
    plt.savefig(save_filename)
    plt.close()


def plot_15s(input_filename, save_filename, wiener=False):
    data_input = np.loadtxt(input_filename, delimiter=",")

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    if wiener:
        ax.plot(data_input[:997], color="orangered")
        # plt.ylim(-0.002, 0.002)
    else:
        ax.plot(data_input[:997])
    ax.grid(color="gray", linestyle="--")
    plt.xticks([0, 332, 665, 997])
    plt.savefig(save_filename)
    plt.close()


def plot_full(input_filename, save_filename, wiener=False):
    data_input = np.loadtxt(input_filename, delimiter=",")

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    if wiener:
        ax.plot(data_input, color="orangered")
    else:
        ax.plot(data_input)
    ax.grid(color="gray", linestyle="--")
    plt.xticks([0, 332, 665, 997, 1330, 1663, 1995])
    plt.savefig(save_filename)
    plt.close()


def visualize_pulse(pulse1, peak1_index, peak2_index, save_filename, wiener=False, part=False):
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

    if part:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=(14, 6))

    ax = fig.add_subplot(111)

    x = np.linspace(0, pulse1.shape[0], pulse1.shape[0])

    if part:
        if wiener:
            ax.plot(pulse1[:360], color='orangered')
        else:
            ax.plot(pulse1[:360])
        plt.xticks([0,60, 120, 180, 240,300,360])
        
        peak1_index = peak1_index[peak1_index < 360] 
        peak2_index = peak2_index[peak2_index < 360] 
     
        
    else:
        if wiener:
            ax.plot(x, pulse1, color='orangered')
        else:
            ax.plot(x, pulse1)
        plt.xticks([0, 332, 665, 997, 1330, 1663, 1995])
    # ax.grid(color="gray", linestyle="--")
    ax.scatter(x[peak1_index], pulse1[peak1_index], marker='x', color='green', s=250)
    ax.scatter(x[peak2_index], pulse1[peak2_index], marker='x', color='green', s=250)
    plt.savefig(save_filename)
    plt.close()


def main():
    
    INPUT_DIR ='/Users/masayakinefuchi/脈波推定/imageSensing/RGB_pulse/program/_skinColorSeparation/result/' 
    OUTPUT_DIR ='/Users/masayakinefuchi/脈波推定/imageSensing/RGB_pulse/program/_plotPredictPulse/result/'
    subject ='ayumu'
    normalized = False
    log_space = False
    sample_rate = 60
    # wavelength_lists = [["800", "735"], ["865", "735"], ["865", "800"], ["930", "735"], ["930", "800"], ["930", "865"]]
    
    s = ""
    if log_space:
        s += "log-space-"
    if normalized:
        s += "normalized-"
    # s += "ver2-"
    pulse_dir = INPUT_DIR
    pulse_previous_filename = pulse_dir + subject+".csv"

    pulse_previous = np.loadtxt(pulse_previous_filename, delimiter=",")
        
    save_previousfilename     = OUTPUT_DIR+ subject+"_predict_hemoglobin.csv"
    save_previous_dt_filename = OUTPUT_DIR+ subject+"_predict_hemoglobin_dt.csv"
    save_previous_bp_filename = OUTPUT_DIR+ subject+"_predict_hemoglobin_bp.csv"
    
    save_previous_dt_img = OUTPUT_DIR + subject+"_predict_hemoglobin.png"
    save_previous_dt_img = OUTPUT_DIR + subject+"_predict_hemoglobin_dt.png"
    save_previous_dt_img_part = OUTPUT_DIR +subject+ "_predict_hemoglobin_dt_part.png"
    save_previous_bp_img_part = OUTPUT_DIR +subject+ "_predict_hemoglobin_bp_part.png"
    save_previous_bp_img_full = OUTPUT_DIR + subject+"_predict_hemoglobin_bp_full.png"
    save_previous_bp_img_15s = OUTPUT_DIR + subject+"_predict_hemoglobin_bp_15s.png"

        

    pulse_previous_dt, pulse_previous_bp, previous_peak1_index, previous_peak2_index = preprocess_pulse(pulse_previous, sample_rate)
        
  
    np.savetxt(save_previous_dt_filename, pulse_previous_dt, delimiter=",")
    np.savetxt(save_previous_bp_filename, pulse_previous_bp, delimiter=",")
        
    plot_part(save_previous_dt_filename, save_previous_dt_img_part)
    plot_full(save_previous_dt_filename, save_previous_dt_img)
    plot_part(save_previous_bp_filename, save_previous_bp_img_part)
    plot_full(save_previous_bp_filename, save_previous_bp_img_full)
    plot_15s(save_previous_bp_filename, save_previous_bp_img_15s)
   


    save_previous_bp_marked = OUTPUT_DIR + subject +"_predict_hemoglobin_bp_marked.png"
    save_previous_bp_marked_part =OUTPUT_DIR + subject +"_predict_hemoglobin_bp_marked_part.png"
        

    visualize_pulse(pulse_previous_bp, previous_peak1_index, previous_peak2_index, save_previous_bp_marked)
    visualize_pulse(pulse_previous_bp, previous_peak1_index, previous_peak2_index, save_previous_bp_marked_part, part=True)
      

    ibi1, pulse_rate1, frq1, psd1 = calculate_hrv2(previous_peak1_index, previous_peak2_index, sample_rate)


        

if __name__ == "__main__":
    main()
