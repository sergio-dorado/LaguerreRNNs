import control as ctrl
from .legendre_pade import *
from nengolib.signal import LinearSystem

def LaguerreDelay(theta, order, a = 0.9, dt = 1):
    
    n = _check_order(order)
    
    # Computing the Pade approximation in CT (using the LMU approximation)
    A, B, C, D = legendre_pade(theta, order)
    ss_pade = ctrl.StateSpace(A, B, C, D)
    
    # Impulse response
    t_k = np.arange(0, 2*theta, dt)
    _, h = ctrl.impulse_response(ss_pade, t_k) 
    
    # Computing the laguerre Network approximation in state-space
    A, L0 = laguerre_network(a = a, N = order)
    c, L = laguerre_coordinates(h, A, L0, t_k) # TODO: is this really required?
    
    L0 = np.reshape(L0, (L0.shape[0], 1))

    return LinearSystem((A, L0, c.T, 0), analog = False)