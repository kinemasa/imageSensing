import numpy as np



def make_moment(K,res):
    M = np.zeros((K+1, K+1))
    
    for m1 in range(K+1):
        for m2 in range(K+1 - m1):
            E12 = np.mean((res[0, :]**m1) * (res[1, :]**m2))
            E1E2 = np.mean(res[0, :]**m1) * np.mean(res[1, :]**m2)
            M[m1, m2] = E12 - E1E2
            
    return M