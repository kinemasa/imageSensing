import numpy as np
import _gfunkT as g

def cost_gmm(K, M):
    CostGMM = 0

    for a1 in range(K+1):
        for a2 in range(K+1 - a1):
            for b1 in range(K+1):
                for b2 in range(K+1 - b1):
                    CostGMM += g.GfuncT(a1, a2, b1, b2) * M[a1, a2] * M[b1, b2]

    return CostGMM