import numpy as np
import scipy.linalg as linalg

def laguerre_network_dt(a, N, n_inputs = 1):
    v = np.zeros(shape = (N,), dtype = np.float64)
    L0 = np.ones(shape = (N, n_inputs), dtype = np.float64)
    
    # First column of the initial condition matrix
    L0_col = np.zeros(shape = (N, 1), dtype = np.float64)
    
    v[0] = a
    L0_col[0, 0] = 1
    
    for k in range(1,N):
        v[k] = np.power(-a, k+1-2)*(1-np.power(a, 2))
        L0_col[k, 0] = np.power(-a, k)
        
    L0_col *= np.sqrt(1-np.power(a,2))
    L0 = L0_col * L0
    
    # The A matrix is lower triangular and Toeplitz. These facts are used to construct it directly.
    first_row = np.zeros(shape = v.shape)
    first_row[0] = v[0]
    
    # Using the Toeplitz function from SciPy
    A_l = linalg.toeplitz(v, first_row)
    
    return A_l, L0