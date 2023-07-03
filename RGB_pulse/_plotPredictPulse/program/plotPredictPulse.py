import time
import math
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.sparse import eye, spdiags
import matplotlib.pyplot as plt
from pathlib import Path

from _bandpass import bandpass_filter_pulse
from _deTrending import detrend_pulse
from _interplolate_nan import interpolate_nan
from _ibi_to_flex import psd_to_frqdomain_feature
from _ibi_to_time import ibi_to_timedomain_feature
from _calculate_hrv2 import calculate_hrv2
from _detect_peak import detect_pulse_peak

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
    # デトレンド / 脈波の終端切り捨てが発生することに注意
    pulse_dt = detrend_pulse(pulse, sample_rate)
  
    # バンドパスフィルタリング / [0.75, 5.0]
    band_width = [0.75, 4.0]
    pulse_bp = bandpass_filter_pulse(pulse_dt, band_width, sample_rate)

    # ピーク検出
    peak1_index, peak2_index = detect_pulse_peak(pulse_bp, sample_rate)

    return pulse_dt, pulse_bp, peak1_index, peak2_index

def plot_part(input_filename, save_filename):
    data_input = np.loadtxt(input_filename, delimiter=",")

    fig = plt.figure()
    ax = fig.add_subplot(111)

 
    ax.plot(data_input[:360])
    ax.grid(color="gray", linestyle="--")
    plt.xticks([60, 120, 180, 240, 300, 360])
    plt.savefig(save_filename)
    plt.close()


def plot_15s(input_filename, save_filename):
    data_input = np.loadtxt(input_filename, delimiter=",")

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

   
    ax.plot(data_input[:997])
    ax.grid(color="gray", linestyle="--")
    plt.xticks([0, 332, 665, 997])
    plt.savefig(save_filename)
    plt.close()


def plot_full(input_filename, save_filename):
    data_input = np.loadtxt(input_filename, delimiter=",")

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)


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
    if part:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=(14, 6))

    ax = fig.add_subplot(111)

    x = np.linspace(0, pulse1.shape[0], pulse1.shape[0])

    if part:
        if wiener:
            ax.plot(pulse1[60:360], color='orangered')
        else:
            ax.plot(pulse1[60:360])
        plt.xticks([0,60,120, 180, 240, 300])
        
        peak1_index = peak1_index[peak1_index < 360] 
      
        peak2_index = peak2_index[peak2_index < 360] 
    
     
        
    else:
        
        ax.plot(x, pulse1)
        plt.xticks([0, 332, 665, 997, 1330, 1663, 1995])
    # ax.grid(color="gray", linestyle="--")
    # ax.scatter(x[peak1_index], pulse1[peak1_index], marker='x', color='green', s=250)
    # ax.scatter(x[peak2_index], pulse1[peak2_index], marker='x', color='green', s=250)
    plt.savefig(save_filename)
    plt.close()


def main():
    ##directory
    path = Path(__file__).parents[2]
    INPUT_DIR =str(path /"_skinColorSeparation"/"result")+"/"
    OUTPUT_DIR =str(path /"_plotPredictPulse"/"result")+"/"
    
    ##dataName
    subject ='ayumu2-open'
   ##sample_rate 
    sample_rate = 60
    
    
    ##importData
    pulse_dir = INPUT_DIR
    pulse_filename = pulse_dir + subject+".csv"

    pulse_previous = np.loadtxt(pulse_filename, delimiter=",")
        
    save__filename     = OUTPUT_DIR+ subject+"_predict_hemoglobin.csv"
    save_dt_filename = OUTPUT_DIR+ subject+"_predict_hemoglobin_dt.csv"
    save_bp_filename = OUTPUT_DIR+ subject+"_predict_hemoglobin_bp.csv"
    
    save_dt_img = OUTPUT_DIR + subject+"_predict_hemoglobin.png"
    save_dt_img = OUTPUT_DIR + subject+"_predict_hemoglobin_dt.png"
    save_dt_img_part = OUTPUT_DIR +subject+ "_predict_hemoglobin_dt_part.png"
    save_bp_img_part = OUTPUT_DIR +subject+ "_predict_hemoglobin_bp_part.png"
    save_bp_img_full = OUTPUT_DIR + subject+"_predict_hemoglobin_bp_full.png"
    save_bp_img_15s = OUTPUT_DIR + subject+"_predict_hemoglobin_bp_15s.png"

        

    pulse_dt, pulse_bp, peak1_index, peak2_index = preprocess_pulse(pulse_previous, sample_rate)
        
  
    np.savetxt(save_dt_filename, pulse_dt, delimiter=",")
    np.savetxt(save_bp_filename, pulse_bp, delimiter=",")
        
    plot_part(save_dt_filename, save_dt_img_part)
    plot_full(save_dt_filename, save_dt_img)
    plot_part(save_bp_filename, save_bp_img_part)
    plot_full(save_bp_filename, save_bp_img_full)
    plot_15s(save_bp_filename, save_bp_img_15s)
   


    save_bp_marked = OUTPUT_DIR + subject +"_predict_hemoglobin_bp_marked.png"
    save_bp_marked_part =OUTPUT_DIR + subject +"_predict_hemoglobin_bp_marked_part.png"
        

    visualize_pulse(pulse_bp, peak1_index, peak2_index, save_bp_marked)
    visualize_pulse(pulse_bp, peak1_index, peak2_index, save_bp_marked_part, part=True)
      

    # ibi1, pulse_rate1, frq1, psd1 = calculate_hrv2(peak1_index, peak2_index, sample_rate)


        

if __name__ == "__main__":
    main()
