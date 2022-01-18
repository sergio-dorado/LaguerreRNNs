import numpy as np
import control as ctrl

import numpy as np
import control as ctrl

def ladder_network(n_inputs, n_outputs, max_delay):

    num_mat = []
    den_mat = []
    
    if n_outputs == 1:
        num_mat = np.array([1])
        den_mat = np.array([1, 0])
        for _ in range(max_delay):
            den_mat = np.convolve(den_mat, np.array([1, 0]))
    else:
        for row in range(n_inputs):
            num_row = []
            den_row = []
            for col in range(n_outputs):
                if row == col:
                    den = np.array([1, 0])
                    for _ in range(1, col*int(max_delay/(n_outputs - 1))):
                        den = np.convolve(den, np.array([1, 0]))
                    num_row.append(np.array([1]))
                    den_row.append(den)
                else:
                    num_row.append(np.array([0]))
                    den_row.append(np.array([1]))

            num_mat.append(num_row)
            den_mat.append(den_row)

    G_tf_mat = ctrl.TransferFunction(num_mat, den_mat, True)

    ss_G = ctrl.tf2ss(G_tf_mat)

    A, B, C, D = np.asarray(ss_G.A), np.asarray(ss_G.B), np.asarray(ss_G.C), np.asarray(ss_G.D)
    
    return A, B, C, D