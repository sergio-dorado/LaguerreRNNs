from .cells import *

import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc

class BRC_SysID:

    def __init__(self, input_shape, n_outputs, dataset,
                 output_dim, build = True):

        # Setting the dataset and the model name
        self.dataset = dataset
        self.model_name = "BRC"

        # Getting input shape data
        self.length = input_shape[0]
        self.n_features = input_shape[-1]

        # Getting the number of outputs
        self.n_outputs = n_outputs

        ####################################
        ### ARCHITECTURE HYPERPARAMETERS ###
        ####################################

        self.output_dim = output_dim

        # Building the model during instantiation
        if build == True:
            self.model = self.build_model()
            # Saving initial model
            self.init_path = os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_model_{self.dataset}_init.hdf5'))
            self.model.save_weights(self.init_path)

    #################################################################
    ########################## BUILDING MODEL #######################
    #################################################################
    # Please modify this method ONLY to vary the hyperparameters of the network

    def build_model(self):

        # Setting seed
        seed = 42
        tf.random.set_random_seed(seed)
        np.random.seed(seed)

        # Creating model
        model = Sequential()

        # Setting architecture for classification task
        model.add(RNN(BistableRecurrentCellLayer(output_dim = self.output_dim),
              input_shape = (self.length, self.n_features),
             return_sequences = True))
        model.add(TimeDistributed(Dense(self.n_outputs)))

        # Setting compiler and plotting model summary
        model.compile(optimizer = "adam", loss = "mean_squared_error", metrics = [tf.keras.metrics.MeanAbsoluteError()])
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

    def fit(self, x_all, y_all, x_valid, y_valid,
            batch_size = 100, verbose = 1, patience = 50, n_epochs = 2000, k_fold = True, n_splits = 5,
            x_train = None, y_train = None, x_test = None, y_test = None):

        self.batch_size = batch_size
        self.verbose = verbose
        self.patience = patience
        self.n_epochs = n_epochs
        self.k_fold = k_fold
        self.n_splits = n_splits

        # Creating path for model results on a given dataset
        if not os.path.exists(os.path.abspath(os.path.join('results', self.dataset))):
            os.mkdir(os.path.abspath(os.path.join('results', self.dataset)))

        # Only for k-Fold validation
        if k_fold == True:
            # Metrics for performance evaluation
            mse_per_fold = []
            mae_per_fold = []
            epoch_per_fold = []
            duration_per_fold = []

            kf = KFold(n_splits = self.n_splits)

        start_time_all = time.time()

        if k_fold == True: # Perform k-fold cross-validation

            for n_fold, (train, test) in enumerate(kf.split(x_all, y_all)):

                self.model.load_weights(self.init_path)

                # Path to save the best model
                file_path = os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_model_{self.dataset}_{n_fold + 1}.hdf5'))

                # Callbacks for training algorithm
                self.callbacks = [
                    ModelCheckpoint(filepath = file_path, monitor = 'val_mean_absolute_error', save_best_only = True, mode = 'min', verbose = self.verbose),
                    EarlyStopping(monitor = 'val_mean_absolute_error', patience = self.patience, mode = 'min'),
                    ReduceLROnPlateau(monitor = 'val_mean_absolute_error', factor = 0.5, patience = self.patience, min_lr = 0.0001)]

                # Starting measuring fold training time
                start_time_fold = time.time()

                print(f"\nFold: {n_fold + 1}/{self.n_splits}")

                # Command training
                hist = self.model.fit(x_all[train], y_all[train],
                                      batch_size = self.batch_size,
                                      epochs = self.n_epochs,
                                      verbose = self.verbose,
                                      validation_data = (x_valid, y_valid),
                                      callbacks = self.callbacks)

                # Training time of each fold
                duration_fold = time.time() - start_time_fold
                # Adding training time to results for each fold
                duration_per_fold.append(duration_fold)

                # Getting training results for current fold
                df_results = pd.DataFrame(hist.history)
                df_results.to_csv(os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_results_{n_fold + 1}.csv')))

                # Computing epoch for best model (on MSE)
                epoch_best = np.argmin(df_results['loss'].values)
                # Determining MSE
                mse = df_results['loss'][epoch_best]

                # Computing epoch for best model (on MAE)
                epoch_best = np.argmin(df_results['mean_absolute_error'].values)
                # Determining MAE
                mae = df_results['mean_absolute_error'][epoch_best]

                epoch_per_fold.append(epoch_best)
                mse_per_fold.append(mse)
                mae_per_fold.append(mae)

                ################################
                #### COMPUTING PREDICTIONS ####
                ################################

                # Loading best model
                self.model.load_weights(file_path)

                # Computing model predictions
                Y_predict_fold = self.model.predict(x_all[test])

                # Saving output and ground truth
                np.savez(os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_Y_predict_{n_fold + 1}.npz')), Y_predict_fold)
                # Saving ground truth
                np.savez(os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_Y_ground_truth_{n_fold + 1}.npz')), y_all[test])

            # Training time for all folds
            duration_all = time.time() - start_time_all
            print(f"Total Training Time: {duration_all:.2f} s")

            # Exporting results per fold
            df_results_folds = pd.DataFrame(columns = ['Epoch_Best', 'MSE', 'MAE', 'Duration'])
            df_results_folds['Epoch_Best'] = epoch_per_fold
            df_results_folds['MSE'] = mse_per_fold
            df_results_folds['MAE'] = mae_per_fold
            df_results_folds['Duration'] = duration_per_fold

            # Saving as .csv
            df_results_folds.to_csv(os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_results_k-Folds.csv')))

            tf.keras.backend.clear_session()

        else:
            # Path to save the best model
            file_path = os.path.abspath(os.path.join('results', self.dataset, f'{self.dataset}-{self.model_name}.hdf5'))

            # Callbacks for training algorithm
            self.callbacks = [
                ModelCheckpoint(filepath = file_path, monitor = 'val_mean_absolute_error', save_best_only = True, mode = 'min', verbose = self.verbose),
                EarlyStopping(monitor = 'val_mean_absolute_error', patience = self.patience, mode = 'min'),
                ReduceLROnPlateau(monitor = 'val_mean_absolute_error', factor = 0.5, patience = self.patience, min_lr = 0.0001)]

            # Starting measuring training time
            start_time = time.time()

            # Command training
            hist = self.model.fit(x_train, y_train,
                                  batch_size = self.batch_size,
                                  epochs = self.n_epochs,
                                  verbose = self.verbose,
                                  validation_data = (x_valid, y_valid),
                                  callbacks = self.callbacks)

            # Total training time
            duration = time.time() - start_time
            print(f"Total Training Time: {duration:.2f} s")

            # Store training results
            df_results = pd.DataFrame(hist.history)

            # Computing epoch for best model (MSE)
            epoch_best =  np.argmin(df_results['loss'].values)
            # Determining MSE
            mse = df_results['loss'][epoch_best]

            # Computing epoch for best model (MAE)
            epoch_best =  np.argmin(df_results['mean_absolute_error'].values)
            # Determining MAE
            mae = df_results['mean_absolute_error'][epoch_best]

            df_results['mae_best'] = mae
            df_results['mse_best'] = mse
            df_results['epoch_best'] = epoch_best

            df_results.to_csv(os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_results.csv')))

            ################################
            #### COMPUTING PREDICTIONS ####
            ################################

            # Loading best model
            self.model.load_weights(file_path)

            # Computing model predictions
            Y_predict = self.model.predict(x_test)

            # Saving output and ground truth
            np.savez(os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_Y_predict.npz')), Y_predict)
            # Saving ground truth
            np.savez(os.path.abspath(os.path.join('results', self.dataset, f'{self.model_name}_Y_ground_truth.npz')), y_test)

            # Closing TF session
            tf.keras.backend.clear_session()
