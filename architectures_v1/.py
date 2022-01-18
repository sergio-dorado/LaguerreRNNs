from .cells import *

import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc

class BRC_Classifier:
    
    def __init__(self, x_all, y_all, dataset, build = True):
        
        # Setting the dataset and the model name
        self.dataset = dataset
        self.model_name = "BRC"
        
        # Determining the length (i.e., number of samples) and the number of features (dimensionality) of the dataset for the task
        if len(x_all.shape) == 2:
            self.length = x_all.shape[1]
            self.n_features = 1
        else:
            self.length = x_all.shape[1]
            self.n_features = x_all.shape[-1]
        
        # Getting the number of classes
        self.n_classes = to_categorical(y_all).shape[-1]
            
        # Building the model during instantiation
        if build == True:
            self.model = self.build_model()
            
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
        model.add(RNN(BistableRecurrentCellLayer(output_dim = 212),
                      input_shape = (None, self.n_features), return_sequences = False))
        model.add(Dense(self.n_classes, activation = "softmax"))
        
        # Setting compiler and plotting model summary
        model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
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
        if not os.path.exists(os.path.abspath(os.path.join('models', self.dataset))):
            os.mkdir(os.path.abspath(os.path.join('models', self.dataset)))

        # Only for k-Fold validation
        if k_fold == True:
            # Metrics for performance evaluation
            acc_per_fold = []
            epoch_per_fold = []
            loss_per_fold = []
            rec_per_fold = []
            prec_per_fold = []
            f1_per_fold = []
            duration_per_fold = []
            
            kf = KFold(n_splits = self.n_splits)
            sdk = StratifiedKFold(n_splits = self.n_splits, random_state = 42, shuffle = True)
        
        # Metrics for binary classification
        if self.n_classes == 2:
            auroc_per_fold = []
            fpr_per_fold = []
            tpr_per_fold = []
            df_roc_curve = pd.DataFrame(columns = range(1, self.n_splits))
            # Array to compute the mean false positive rate (100 points)
            mean_fpr = np.linspace(0, 1, 100)
        
        start_time_all = time.time()
        
        if k_fold == True: # Perform k-fold cross-validation
        
            for n_fold, (train, test) in enumerate(sdk.split(x_all, y_all)):

                # Path to save the best model
                file_path = os.path.abspath(os.path.join('models', self.dataset, f'{self.model_name}_model_{self.dataset}_{n_fold + 1}.hdf5'))

                # Callbacks for training algorithm
                self.callbacks = [
                    ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'min', verbose = self.verbose), 
                    EarlyStopping(monitor = 'val_loss', patience = self.patience, mode = 'min'), 
                    ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = self.patience, min_lr = 0.0001)]
                
                # Starting measuring fold training time
                start_time_fold = time.time()
                
                print(f"\nFold: {n_fold + 1}/{self.n_splits}")
                
                # Command training
                hist = self.model.fit(x_all[train], to_categorical(y_all[train]), 
                                      batch_size = self.batch_size, 
                                      epochs = self.n_epochs,
                                      verbose = self.verbose, 
                                      validation_data = (x_valid, to_categorical(y_valid)),
                                      callbacks = self.callbacks)
                
                # Training time of each fold
                duration_fold = time.time() - start_time_fold
                # Adding training time to results for each fold 
                duration_per_fold.append(duration_fold) 

                # Store training results
                df_results = pd.DataFrame(hist.history)
                df_results.to_csv(os.path.abspath(os.path.join('models', self.dataset, f'{self.model_name}_results_{n_fold + 1}.csv')))

                # Loading best model
                self.model.load_weights(file_path)

                # Evaluating performance using corresponding performance metrics
                scores = self.model.evaluate(x_all[test], to_categorical(y_all[test])) # Testing loss and testing accuracy
                loss_per_fold.append(scores[0])
                acc_per_fold.append(scores[1])
                
                # Epoch of the best model
                epoch_per_fold.append(np.argmin(hist.history['val_loss'])) # Epoch for the best model
                
                # Computing predictions
                y_pred_test = np.argmax(self.model.predict(x_all[test]), axis = 1)
                
                # Getting performance scores per fold
                f1_per_fold.append(f1_score(y_all[test], y_pred_test, average = 'macro'))
                rec_per_fold.append(recall_score(y_all[test], y_pred_test, average = 'macro'))
                prec_per_fold.append(precision_score(y_all[test], y_pred_test, average = 'macro'))
                
                # Inside this conditional, the binary classification metrics are evaluated
                if self.n_classes == 2:
                    
                    # Computing logits
                    y_logits = self.model.predict(x_all[test])

                    # Computing scores for the positive class only (taking position '1' as the positive class)
                    y_scores = y_logits[:, 1]
                    
                    # Computing roc_curve (false positive and true positive rate)
                    _fpr, _tpr, _ = roc_curve(y_all[test], y_scores)

                    # Calculating the tpr in exact 100 points using linear interpolation
                    interp_tpr = np.interp(mean_fpr, _fpr, _tpr)
                    interp_tpr[0] = 0.0

                    # Saving fpr and tpr per fold
                    df_roc_curve[n_fold + 1] = interp_tpr

                    # AUC score
                    auroc_score = auc(_fpr, _tpr)
                    auroc_per_fold.append(auroc_score)

            if self.n_classes == 2:
                # Computing standard deviation
                std_dev_tpr = df_roc_curve.std(axis = 1)
                # Computing and adding mean into the dataframe
                df_roc_curve['mean_tpr'] = df_roc_curve.mean(axis = 1)
                # Adding standard deviation
                df_roc_curve['std_tpr'] = std_dev_tpr
                # Adding fpr to the dataframe
                df_roc_curve['fpr'] = mean_fpr
            
            # Training time for all folds
            duration_all = time.time() - start_time_all
            print(f"Total Training Time: {duration_all:.2f} s")
            
            # Storing performance data in a dataframe
            df_scores = pd.DataFrame(columns = ['F1', 'Loss', 'Accuracy', 'Precision', 'Recall', 'Duration'])
            df_scores['F1'] = f1_per_fold
            df_scores['Loss'] = loss_per_fold
            df_scores['Accuracy'] = acc_per_fold
            df_scores['Precision'] = prec_per_fold
            df_scores['Recall'] = rec_per_fold
            df_scores['Duration'] = duration_per_fold
            
            # Saving AUROC for binary classification
            if self.n_classes == 2: df_scores['AUROC'] = auroc_per_fold
            
            # Saving scores
            df_scores.to_csv(os.path.abspath(os.path.join('models', self.dataset, f'{self.model_name}_results_k-Folds.csv')))
            
            # Saving information to plot roc curve for binary classification
            if self.n_classes == 2:
                df_roc_curve.to_csv(os.path.abspath(os.path.join('models', self.dataset, f'{self.model_name}_roc_curve_k_Folds.csv')))

            tf.keras.backend.clear_session()
            
        else: 
            
            if not self.dataset == 'psMNIST':
                # Splitting data into training and testing (0.2 proportion) 
                # recall that validation data was already passed
                # Training and testing data are given for psMNIST
                x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = 0.2)
            
            # Path to save the best model
            file_path = os.path.abspath(os.path.join('models', self.dataset, f'{self.dataset}-{self.model_name}.hdf5'))

            # Callbacks for training algorithm
            self.callbacks = [
                ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'min', verbose = self.verbose), 
                EarlyStopping(monitor = 'val_loss', patience = self.patience, mode = 'min'), 
                ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = self.patience, min_lr = 0.0001)]

            # Starting measuring training time
            start_time = time.time()

            # Command training
            hist = self.model.fit(x_train, to_categorical(y_train), 
                                  batch_size = self.batch_size, 
                                  epochs = self.n_epochs,
                                  verbose = self.verbose, 
                                  validation_data = (x_valid, to_categorical(y_valid)),
                                  callbacks = self.callbacks)
            
            # Total training time
            duration = time.time() - start_time
            print(f"Total Training Time: {duration:.2f} s")

            # Store training results
            df_results = pd.DataFrame(hist.history)
            df_results.to_csv(os.path.abspath(os.path.join('models', self.dataset, f'{self.model_name}_results.csv')))

            # Loading best model
            self.model.load_weights(file_path)
            
            # Computing testing loss and accuracy
            scores = self.model.evaluate(x_test, to_categorical(y_test))
            
            # Computing logits
            y_logits = self.model.predict(x_test)
            y_preds = np.argmax(y_logits, axis = 1)
            
            # Dataframe to compute scores
            df_scores = pd.DataFrame()
            
            # Populating results from TensorFlow arrays
            df_scores['Loss'] = scores[0]
            df_scores['Accuracy'] = scores[1]
            
            # Populating scores from scikit-learn functions
            df_scores['Precision'] = precision_score(y_test, y_preds, average = 'macro')
            df_scores['Recall'] = recall_score(y_test, y_preds, average = 'macro')
            df_scores['F1'] = f1_score(y_test, y_preds, average = 'macro')
            
            if self.n_classes == 2:
                # Computation of AUROC - ROC curve
                _fpr, _tpr, _ = roc_curve(y_test, y_logits[:,1])
                
                # Calculating the tpr in exact 100 points using linear interpolation
                interp_tpr = np.interp(mean_fpr, _fpr, _tpr)
                interp_tpr[0] = 0.0
                
                # Saving fpr and tpr per fold
                df_roc_curve['fpr'] = mean_fpr
                df_roc_curve['tpr'] = interp_tpr

                # AUC score
                auroc_score = auc(_fpr, _tpr)
                df_scores['AUROC'] = auroc_score
            
            # Exporting results
            df_scores.to_csv(os.path.abspath(os.path.join('models', self.dataset, f'{self.model_name}_results.csv')))
            
            # Exporting ROC results for binary classification
            if self.n_classes == 2:
                df_roc_curve.to_csv(os.path.abspath(os.path.join('models', self.dataset, f'{self.model_name}_roc_curve.csv')))

            tf.keras.backend.clear_session()