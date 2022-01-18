import numpy as np

def laguerre_network_ct(p, N, n_inputs = 1):
    
    # Column and row indices
    j, i = np.meshgrid(np.arange(1, N + 1), np.arange(1, N + 1))
    
    # Creating `A` matrix
    A = np.where(j < i, -2*p, np.where(i == j, -p, 0))
    
    # Creating `L0` vector (vector of initial conditions)
    L0 = np.sqrt(2*p)*np.ones(shape = (N, n_inputs))
    
    # Changing numerical type of A to increase numerical precision
    A = A.astype('float64') 
    
    return A, L0