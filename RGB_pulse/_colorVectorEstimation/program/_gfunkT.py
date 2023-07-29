import numpy as np
import math

def factT(n):
    return math.factorial(n)

def GfuncT(ga1, ga2, gb1, gb2):
    Sigma = 1

    G = 1
    if (ga1 + gb1) % 2 == 0:
        k = (ga1 + gb1) // 2
        J2k = (factT(2 * k) * (2 * np.pi)**(1/2)) / (((4**k) * factT(k)) * (Sigma**(2 * k - 1)))

        sg = ((-1)**((ga1 - gb1) // 2) * J2k) / (factT(ga1) * factT(gb1))
        G *= sg
    else:
        G = 0

    if (ga2 + gb2) % 2 == 0:
        k = (ga2 + gb2) // 2
        J2k = (factT(2 * k) * (2 * np.pi)**(1/2)) / (((4**k) * factT(k)) / (Sigma**(2 * k - 1)))

        sg = ((-1)**((ga2 - gb2) // 2) * J2k) / (factT(ga2) * factT(gb2))
        G *= sg
    else:
        G = 0

    y = G
    return y