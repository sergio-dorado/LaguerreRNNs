import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import os
import random

from sklearn.model_selection import train_test_split

import sys

from utils import *
from architectures_v1 import *
from datasets import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN, TimeDistributed, GRU, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Definition of the pendulum dynamic equations
def pend(x, t, u, b, c):
    x1, x2 = x
    dxdt = [x2, -b*x2 - c*np.sin(x1) + u]
    return dxdt

# Definition of the fluid flow dynamic equations
def fluid_flow(x, t, u, mu, omega, A, l):
    x1, x2, x3 = x
    dxdt = [mu*x1 - omega*x1 + A*x1*x3 + u,
           omega*x1 - omega*x2 + A*x2*x3,
           -l*(x3 - x1**2 - x2**2)]
    return dxdt

# Function to generate input/output data
def generate_data_IO(t, n_instances, n_states, dyn_system = "Pendulum", **kwargs):

    if dyn_system == "Pendulum":
        if 'b' not in kwargs.keys() or 'c' not in kwargs.keys():
            b = 0.5
            c = 1.0
        else:
            b = kwargs['b']
            c = kwargs['c']
    elif dyn_system == "FluidFlow":
        if 'mu' not in kwargs.keys() or 'omega' not in kwargs.keys() or 'A' not in kwargs.keys() or 'l' not in kwargs.keys():
            mu = 0.1
            omega = 1.0
            A = -0.1
            l = 10
        else:
            mu = kwargs['mu']
            omega = kwargs['omega']
            A = kwargs['A']
            l = kwargs['l']

    # Creating containers for the inputs and the outputs (i.e., random inputs and states) given a number of instances
    Y = np.zeros(shape = (n_instances, t.shape[0], n_states))

    # Container for inputs to the NN
    X = np.zeros(shape = (n_instances, t.shape[0], n_states))

    for n_inst in range(n_instances):

        u = np.random.rand(t.shape[0], )

        # Creating random initial state
        x0 = np.random.rand(n_states, )

        X[n_inst, 0, :] = x0
        Y[0, 0, :] = x0

        for k in range(len(t) - 1):

            # Integration time
            t_int = np.linspace(t[k], t[k+1], 500)

            if dyn_system == "Pendulum":
                # Integration
                x_int = odeint(pend, x0, t_int, args = (u[k], b, c))
            elif dyn_system == "FluidFlow":
                # Integration
                x_int = odeint(fluid_flow, x0, t_int, args = (u[k], mu, omega, A, l))

            # Extracting the outputs
            for state in range(x_int.shape[-1]):
                Y[n_inst, k+1, state] = x_int[-1, state]

            # Updating initial condition for next integration
            x0 = x_int[-1, :]

        # Storing input in container for NN input
        X[n_inst, 1:, 0] = u[:k+1]

    return X, Y

# Start of main program
if __name__ == "__main__":

    # Getting model name and dynamic system
    model_name = sys.argv[1]
    dyn_system = sys.argv[2]

    # Perform k-fold cross-validation
    if sys.argv[3] == 'k-fold':
        k_fold = True
    else:
        k_fold = False

    if model_name not in LIST_OF_ARCHITECTURES_SYSID:
        raise ValueError("Model not defined")

    if dyn_system not in ["Pendulum", "FluidFlow"]:
        raise ValueError("Dynamical system not defined")

    print(f"Model: {model_name}")
    print(f"Dynamical System: {dyn_system}\n")

    # Treating the dynamical system as a dataset (for internal purposes of the RNN)
    dataset = dyn_system

    # Time vector
    t = np.linspace(0, 100, 500)

    if dyn_system == "Pendulum":
        # Parameters of the system
        b = 0.25
        c = 5.00
        # Data generation
        X, Y = generate_data_IO(t = t, n_instances = 500, n_states = 2, dyn_system = dyn_system, b = 0.25, c = 5.00)

        # Shape of the input for the RNN
        input_shape = (t.shape[0], 2)
    else:
        # Parameters of the system
        mu = 0.1
        omega = 1
        A = -0.1
        l = 10
        # Data generation
        X, Y = generate_data_IO(t = t, n_instances = 500, n_states = 3, dyn_system = dyn_system, mu = mu, omega = omega, A = A, l = l)

        # Shape of the input for the RNN
        input_shape = (t.shape[0], 3)

    # Splitting data into training and testing
    n_instances = X.shape[0]
    n_validation = int(0.1*X.shape[0])
    ind_validation = random.sample(range(0, n_instances), n_validation)

    X_valid = X[ind_validation, :, :]
    Y_valid = Y[ind_validation, :, :]

    X_all = np.delete(X, ind_validation, axis = 0)
    Y_all = np.delete(Y, ind_validation, axis = 0)

    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size = 0.2, random_state = 42)

    print(f"X_train: {X_train.shape} - Y_train: {Y_train.shape}")
    print(f"X_test: {X_test.shape} - Y_test: {Y_test.shape}")
    print(f"X_valid: {X_valid.shape} - Y_valid: {Y_valid.shape}")

    # Number of outputs
    n_outputs = Y.shape[-1]

    # Classifier creation
    if model_name == "BRC":
        classifier = BRC_SysID(input_shape = input_shape, n_outputs = n_outputs, dataset = dataset,
                                    output_dim = 400)
    elif model_name == "LMU":
        classifier = LMU_SysID(input_shape = input_shape, n_outputs = n_outputs, dataset = dataset,
                                    units = 212, order = 256, dt = max(t)/len(t))
    elif model_name == "nBRC":
        classifier = nBRC_SysID(input_shape = input_shape, n_outputs = n_outputs, dataset = dataset,
                                    output_dim = 400)
    elif model_name == "Laguerre":
        classifier = Laguerre_SysID(input_shape = input_shape, n_outputs = n_outputs, dataset = dataset,
                                        units = 212, order = 256, variant = 'ct_laguerre', dt = max(t)/len(t))
    elif model_name == "Ladder":
        classifier = Ladder_SysID(input_shape = input_shape, n_outputs = n_outputs, dataset = dataset,
                                      units = 212, max_delay = X_all.shape[1])
    elif model_name == "LSTM":
        classifier = LSTM_SysID(input_shape = input_shape, n_outputs = n_outputs, dataset = dataset,
                                      units = 212)
    elif model_name == "GRU":
        classifier = GRU_SysID(input_shape = input_shape, n_outputs = n_outputs, dataset = dataset,
                                      units = 212)

    ###################################
    ###### TRAINING CLASSIFIER ########
    ###################################
    classifier.fit(X_all, Y_all, X_valid, Y_valid,
                   batch_size = 20, verbose = 1, patience = 50, n_epochs = 2000, k_fold = k_fold, n_splits = 5,
                   x_train = X_train, y_train = Y_train, x_test = X_test, y_test = Y_test)
