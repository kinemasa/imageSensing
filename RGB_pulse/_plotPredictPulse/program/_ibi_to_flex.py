import time
import math
import numpy as np
from scipy.sparse import eye, spdiags



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