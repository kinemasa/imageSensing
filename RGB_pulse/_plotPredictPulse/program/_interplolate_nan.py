import time
import math
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.sparse import eye, spdiags
import matplotlib.pyplot as plt

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