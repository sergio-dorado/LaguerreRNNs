from .cells import *

import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc

class Ladder_Prediction:

    def __init__(self, input_shape, dataset,
                 units, max_delay,
                 kernel_initializer, recurrent_initializer, bias_initializer,
                 optimizer = "adam", build = False):

        # Setting the dataset and the model name
        self.dataset = dataset
        self.model_name = "Ladder"

        # Getting input shape data
        self.length = input_shape[0]
        self.n_features = input_shape[-1]
        self.input_dims = self.n_features

        ####################################
        ### ARCHITECTURE HYPERPARAMETERS ###
        ####################################
        self.units = units
        self.max_delay = max_delay

        # Initializer settings
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        # Optimizer selection
        self.optimizer = optimizer

        # Building the model during instantiation
        if build == True:
            self.model = self.build_model()

    #################################################################
    ########################## BUILDING MODEL #######################
    #################################################################
    # Please modify this method ONLY to vary the hyperparameters of the network

    def build_model(self):

        # Creating model
        model = Sequential()

        # Setting architecture for prediction task
        model.add(RNN(LadderCell(units = self.units,
                             max_delay = self.max_delay,
                             input_dims = self.input_dims,
                             kernel_initializer = self.kernel_initializer,
                             recurrent_initializer = self.recurrent_initializer,
                             bias_initializer = self.bias_initializer),
              input_shape = (self.length, self.n_features),
             return_sequences = False))
        model.add(Dense(self.n_features))

        # Setting compiler and plotting model summary
        model.compile(optimizer = self.optimizer, loss = "mean_squared_error")
        model.summary()

        return model

    #################################################################
    ########################## BUILDING MODEL #######################
    #################################################################
    # Training model
    # See the `README` file for more information.
    #
    # Callbacks were modified from those used in:
    #
    # Voelker, A., Kajić, I., & Eliasmith, C. (2019). Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks.
    # Advances in Neural Information Processing Systems, 15570-15579.

    # Fawaz, H. I., Forestier, G., Weber, J., Idoumghar, L., & Muller, P. A. (2019). Deep learning for time series classification: a review.
    # Data Mining and Knowledge Discovery, 33(4), 917-963.

    def fit(self, train_generator, test_generator, scaler,
            n_seeds = 1, verbose = 1, patience = 50, n_epochs = 2000,
            scaled_train = None, x_test = None):

        self.train_generator = train_generator
        self.test_generator = test_generator
        self.scaler = scaler

        self.scaled_train = scaled_train
        self.x_test = x_test

        self.n_seeds = n_seeds
        self.verbose = verbose
        self.patience = patience
        self.n_epochs = n_epochs

        # Creating path for model results on a given dataset
        if not os.path.exists(os.path.abspath(os.path.join('results', self.dataset))):
            os.mkdir(os.path.abspath(os.path.join('results', self.dataset)))

        start_time_all = time.time()

        # Getting seeds
        seeds = random.sample(range(0, 100), self.n_seeds)
        print(f"seeds: {seeds}")

        for n_seed, seed in enumerate(seeds):

            # Path to save the best model
            file_path = os.path.abspath(os.path.join('results', self.dataset, f'{self.dataset}-{self.model_name}_{n_seed + 1}.hdf5'))

            # Callbacks for training algorithm (only using training set-based metrics)
            self.callbacks = [
                ModelCheckpoint(filepath = file_path, monitor = 'loss', save_best_only = True, mode = 'min', verbose = self.verbose),
                EarlyStopping(monitor = 'loss', patience = self.patience, mode = 'min'),
                ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = self.patience, min_lr = 0.0001)]

            # Setting seeds
            tf.compat.v1.random.set_random_seed(seed)
            np.random.seed(seed)

            # Building model
            self.model = self.build_model()

            # Starting measuring training time
            start_time = time.time()

            # Command training
            hist = self.model.fit_generator(self.train_generator,
                                  epochs = self.n_epochs,
                                  verbose = self.verbose,
                                  callbacks = self.callbacks)

            # Total training time
            duration = time.time() - start_time
            print(f"Total Training Time: {duration:.2f} s - {duration/60:.2f} min - {duration/3600:.2f} h")

            # Store training results
            df_results = pd.DataFrame(hist.history)
            df_results.to_csv(os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_results_{n_seed + 1}.csv')))

            #########################################
            ######## EVALUATING PERFORMANCE #########
            #########################################

            # Loading best model
            self.model.load_weights(file_path)

            # Extracting the number of states from the given data
            n_states = x_test.shape[-1]
            test_prediction = dict.fromkeys(np.arange(1, n_states + 1, 1))

            # Adding an empty list to each state
            for state in test_prediction:
                test_prediction[state] = []

            # Take the last part of the training set as the first input to evaluate the network
            first_eval_batch = scaled_train[-self.length:]
            #print(f'first_eval_batch: {first_eval_batch.shape}')

            # Reshape the first evaluation batch to have the compatible shape (n_minibatches, length, n_features)
            current_batch = first_eval_batch.reshape((1, self.length, self.n_features))
            #print(f'current_batch: {current_batch.shape}')

            # Container for relative prediction error
            error = np.zeros_like(x_test)

            for k in range(len(x_test)):

                # Compute prediction 1 time stamp ahead ([0] is for grabbing just the array values instead of [array])
                current_pred = self.model.predict(current_batch)[0]

                for n_state, state in enumerate(current_pred):
                    test_prediction[n_state + 1].append(state)

                # update batch to now include prediction and drop first value
                current_batch = np.append(current_batch[:, 1: ,:],[[current_pred]], axis = 1)

            df_test_predictions = pd.DataFrame(columns = np.arange(1, n_states + 1, 1))

            for state in test_prediction:
                df_test_predictions[state] = np.asarray(test_prediction[state])

            test_predictions = self.scaler.inverse_transform(df_test_predictions.values)

            # Error computation
            for k in range(1, len(x_test)):
                #print(test_predictions[:k, ].shape)
                error[k] = np.linalg.norm(test_predictions[:k, ] - x_test[:k, ]) / np.linalg.norm(x_test[:k, ])

            df_test_predictions = pd.DataFrame(data = test_predictions)
            df_error = pd.DataFrame(data = error)
            df_x_test = pd.DataFrame(data = x_test)

            # Saving predictions and error
            df_test_predictions.to_csv(os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_predictions_{n_seed + 1}.csv')))
            df_error.to_csv(os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_error_{n_seed + 1}.csv')))
            df_x_test.to_csv(os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_x_test_{n_seed + 1}.csv')))

        tf.keras.backend.clear_session()
