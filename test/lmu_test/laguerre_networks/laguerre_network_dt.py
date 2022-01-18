import numpy as np
import scipy.linalg as linalg

def laguerre_network_dt(a, N):
    v = np.zeros(shape = (N,), dtype = np.float64)
    L0 = np.zeros(shape = (N,), dtype = np.float64)
    
    v[0] = a
    L0[0] = 1
    
    for k in range(1,N):
        v[k] = np.power(-a, k+1-2)*(1-np.power(a, 2))
        L0[k] = np.power(-a, k)
        
    L0 *= np.sqrt(1-np.power(a,2))
    
    # The A matrix is lower triangular and Toeplitz. These facts are used to construct it directly.
    first_row = np.zeros(shape = v.shape)
    first_row[0] = v[0]
    
    # Using the Toeplitz function from SciPy
    A_l = linalg.toeplitz(v, first_row)
    
    return A_l, L0