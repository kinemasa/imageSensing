import numpy as np
import _fmin_Make_M   as m
import _fmin_Cal_Cost_Bruel as C
# グローバル変数


def f_burel(s,sensor,res):
    

    x1 = np.cos(s[0])
    y1 = np.sin(s[0])
    x2 = np.cos(s[1])
    y2 = np.sin(s[1])
    H = np.array([[x1, y1], [x2, y2]])

    res = np.dot(H, sensor)

    # Cost Cal.
    K = 4
    M = m.make_moment(K,res)

    f = C.cost_gmm(K,M)

    return f