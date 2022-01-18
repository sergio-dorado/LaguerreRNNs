import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import os
#os.environ['PYTHONHASHSEED'] = str(0) # Setting global hash seed
import random
#random.seed(0) # Setting seed for random library

from sklearn.model_selection import train_test_split

import sys

from utils import *
from architectures_v1 import *
from datasets import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN, TimeDistributed, GRU, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Definition of the pendulum dynamic equations
def pend(x, t, g, l):
    x1, x2 = x
    dxdt = [x2,
            - (g/l) * np.sin(x1)]
    return dxdt

# Definition of the fluid flow dynamic equations
def fluid_flow(x, t, mu, omega, A, l):
    x1, x2, x3 = x
    dxdt = [mu*x1 - omega*x1 + A*x1*x3,
           omega*x1 - omega*x2 + A*x2*x3,
           -l*(x3 - x1**2 - x2**2)]
    return dxdt

# Function to generate input/output data
def generate_data(dyn_system = "Pendulum", noise = 0.0, theta_0 = 0.8):

    if dyn_system == "Pendulum":

        # Parameters for the pendulum system
        g = 9.8
        l = 1.0

        # Time vector
        t = np.arange(0, 170, 0.1)

        # Vector of initial conditions (angle)
        x0 = np.array([theta_0, 0.0])

        x_int = odeint(pend, x0, t, args = (g, l))

        # Adding noise (if any noise passed)
        x_int = x_int + np.random.standard_normal(x_int.shape) * noise

        return x_int

    elif dyn_system == "FluidFlow":

        # Parameters for the fluid flow system
        mu = 0.1
        omega = 1.0
        A = -0.1
        l = 10

        # Time vector
        t = np.arange(0, 2.0, 0.001)

        # Vector of initial conditions
        x0 = np.array([0.453, 0.931, 0.326])

        # Integrating differential equations
        x_int = odeint(fluid_flow, x0, t, args = (mu, omega, A, l))

        # Add noise
        x_int += np.random.standard_normal(x_int.shape) * noise

        return x_int

# Start of main program
if __name__ == "__main__":

    ###################################
    ##### PROCESSING USER INPUTS ######
    ###################################

    # Getting RNN model

    # Getting dynamical system
    dyn_system = "Pendulum"
    dataset = dyn_system + '_Prediction'

    noise = 0.0

    n_seeds = 1
    theta_0 = 0.8

    ############################################
    ##### GENERATING DATA VIA INTEGRATION ######
    ############################################

    x_int = generate_data(dyn_system = dyn_system, theta_0 = theta_0)

    #####################################################
    ##### SPLITTING DATA INTO TRAINING AND TESTING ######
    #####################################################

    # This maintains the proportion used in the paper:
    # Azencot, O., Erichson, N. B., Lin, V., & Mahoney, M. W. (2020). Forecasting sequential data using consistent Koopman autoencoders. arXiv preprint arXiv:2003.02236.
    n_instances = int((600/1700) * x_int.shape[0])

    # Splitting data into training and testing sets
    x_train = x_int[:n_instances, ]
    x_test = x_int[n_instances:, ]

    # Scaling data with scikit learn scaler
    scaler = MinMaxScaler();
    scaler.fit(x_train);

    scaled_train = scaler.transform(x_train)
    scaled_test = scaler.transform(x_test)

    # Using keras TimeseriesGenerator to generate batches for training and testing

    # Number of features (i.e., number of states)
    n_features = x_int.shape[-1]

    # Length of the output sequences (in number of timesteps)
    length = 64
    # Number of timeseries samples in each batch
    batch_size = 30

    train_generator = TimeseriesGenerator(scaled_train, scaled_train, length = length, batch_size = batch_size)
    test_generator = TimeseriesGenerator(scaled_test, scaled_test, length = length, batch_size = batch_size)

    # Defining shape of the inputs to the RNN
    input_shape = (length, n_features)

    models = ["BRC", "LMU", "nBRC", "Laguerre", "Ladder", "LSTM", "GRU"]

    for model_name in models:

        # Classifier creation
        if model_name == "BRC":
            classifier = BRC_Prediction(input_shape = input_shape, dataset = dataset,
                                        output_dim = 400)
        elif model_name == "LMU":
            classifier = LMU_Prediction(input_shape = input_shape, dataset = dataset,
                                        units = 212, order = 256)
        elif model_name == "nBRC":
            classifier = nBRC_Prediction(input_shape = input_shape, dataset = dataset,
                                        output_dim = 400)
        elif model_name == "Laguerre":
            classifier = Laguerre_Prediction(input_shape = input_shape, dataset = dataset,
                                            units = 212, order = 256, variant = 'ct_laguerre', dt = 1,
                                            kernel_initializer = 'glorot_uniform',
                                            recurrent_initializer = 'glorot_uniform',
                                            bias_initializer = 'zeros',
                                            optimizer = 'Ftrl')
        elif model_name == "Ladder":
            classifier = Ladder_Prediction(input_shape = input_shape, dataset = dataset,
                                          units = 212, max_delay = length,
                                          kernel_initializer = 'glorot_uniform',
                                          recurrent_initializer = 'glorot_uniform',
                                          bias_initializer = 'zeros',
                                          optimizer = 'Ftrl')
        elif model_name == "LSTM":
            classifier = LSTM_Prediction(input_shape = input_shape, dataset = dataset,
                                          units = 212)
        elif model_name == "GRU":
            classifier = GRU_Prediction(input_shape = input_shape, dataset = dataset,
                                          units = 212)

        classifier.fit(train_generator, test_generator, scaler = scaler,
                       n_seeds = n_seeds, verbose = 1, patience = 50, n_epochs = 1,
                       scaled_train = scaled_train, x_test = x_test)
