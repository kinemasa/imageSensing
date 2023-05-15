import time
import math
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


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