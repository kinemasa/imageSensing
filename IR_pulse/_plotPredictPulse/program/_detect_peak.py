import time
import math
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.sparse import eye, spdiags
import matplotlib.pyplot as plt


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
