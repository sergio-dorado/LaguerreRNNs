import numpy as np
from scipy.linalg import expm
from scipy.integrate import trapz

def laguerre_coordinates_ct(A, L0, t, h):
    
    # Creating an array for the Laguerre polynomials
    N, L = A.shape[0], np.zeros(shape = (A.shape[0], t.shape[0]))
    
    L[:,0] = np.reshape(L0, (L0.shape[0],))
    
    # Evaluating Laguerre polynomials up to order N
    for t_k in range(t.shape[0]):
        L[:,t_k] = np.matmul(expm(A*t[t_k]), L[:,0])
    
    # Computing the coordinates
    c = np.zeros(shape = (N,))
    for n in range(N):
        c[n] = trapz(L[n,:]*h, t)
    return c, L