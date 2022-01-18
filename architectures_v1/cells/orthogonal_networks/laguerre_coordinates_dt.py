import numpy as np

def laguerre_coordinates_dt(h, A, L0, t_k, verbose = False):
    
    L = np.zeros(shape = (L0.shape[0], t_k.shape[0]))
    if verbose: print(f"L = {L.shape}")
    
    L[:,0] = L0

    for i in range(1, max(t_k.shape)):
        L[:,i] = np.matmul(A, L[:,i-1])
    if verbose: print(f"L = {L.shape}")
    
    # Coefficients of the Laguerre functions
    c = np.zeros(shape = (L.shape[0],))
    if verbose: print(f"c = {c.shape}")
    
    for n_f in range(L.shape[0]):
        c[n_f] = np.matmul(L[n_f, :], h)
    
    return c, L