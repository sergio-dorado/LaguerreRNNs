import numpy as np

def _check_order(order):
    if order < 1 or not is_integer(order):
        raise ValueError("order (%s) must be integer >= 1" % order)
    return order

def is_integer(obj):
    int_types = (int,)
    return isinstance(obj, int_types + (np.integer,))

def legendre_pade(theta, order, n_inputs = 1):

    q = _check_order(order)
    
    B = np.ones(shape = (order, n_inputs))

    Q = np.arange(q, dtype=np.float64)
    R = (2*Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
    b = (-1.)**Q[:, None] * R
    C = np.ones((1, q))
    D = np.zeros((1, n_inputs))
    
    B = b * B
    
    return A, B, C, D