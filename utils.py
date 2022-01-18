import os
import numpy as np
import random

def train_test_valid_split(dataset):
    
    if dataset == 'psMNIST':
        # Loading dataset
        x_train_load = np.load(os.path.abspath(os.path.join('datasets', dataset, 'x_train.npz')))
        x_test_load = np.load(os.path.abspath(os.path.join('datasets', dataset, 'x_test.npz')))
        x_valid_load = np.load(os.path.abspath(os.path.join('datasets', dataset, 'x_valid.npz')))
        
        # Extracting arrays
        x_train = x_train_load['arr_0']
        x_test = x_test_load['arr_0']
        x_valid = x_valid_load['arr_0']
        
        # Loading targets
        y_train_load = np.load(os.path.abspath(os.path.join('datasets',dataset,'y_train.npz')))
        y_test_load = np.load(os.path.abspath(os.path.join('datasets',dataset,'y_test.npz')))
        y_valid_load = np.load(os.path.abspath(os.path.join('datasets',dataset,'y_valid.npz')))
        
        # Extracting targets
        y_train = y_train_load['arr_0']
        y_test = y_test_load['arr_0']
        y_valid = y_valid_load['arr_0']
        
        # Concatenating training and testing (for k-fold validation)
        x_all = np.concatenate((x_train, x_test), axis = 0)
        y_all = np.concatenate((y_train, y_test), axis = 0)

        print(f"x_all (train + test): {x_all.shape} - y_train: {y_train.shape}")
        print(f"x_train: {x_train.shape} - y_train: {y_train.shape}")
        print(f"x_test: {x_test.shape} - y_test: {y_test.shape}")
        print(f"x_valid: {x_valid.shape} - y_valid: {y_valid.shape}")
        
        # Notice that this output has two additional instances (i.e., training and testing datasets)
        return {'x_all' : x_all, 'y_all' : y_all,
                'x_train': x_train, 'y_train': y_train,
                'x_test' : x_test, 'y_test': y_test,
           'x_valid' :  x_valid, 'y_valid' : y_valid}
    
    x_train_load = np.load(os.path.abspath(os.path.join('datasets', dataset,'x_train.npz')))
    x_test_load = np.load(os.path.abspath(os.path.join('datasets', dataset,'x_test.npz')))

    # Loading input instances
    if len(x_train_load['arr_0'].shape) == 2:
        x_train = np.reshape(x_train_load['arr_0'], [x_train_load['arr_0'].shape[0], x_train_load['arr_0'].shape[1], 1])
        x_test = np.reshape(x_test_load['arr_0'], [x_test_load['arr_0'].shape[0], x_test_load['arr_0'].shape[1], 1])
    else:
        x_train = x_train_load['arr_0']
        x_test = x_test_load['arr_0']
        
    # Loading target labels
    y_train_load = np.load(os.path.abspath(os.path.join('datasets', dataset,'y_train.npz')))
    y_test_load = np.load(os.path.abspath(os.path.join('datasets', dataset,'y_test.npz')))
    
    y_train = y_train_load['arr_0']
    y_test = y_test_load['arr_0']
    
    # Concatenating all 
    x_all = np.concatenate((x_train, x_test), axis = 0)
    y_all = np.concatenate((y_train, y_test), axis = 0)

    n_instances = x_all.shape[0]
    n_validation = int(0.1*n_instances)
    print(f"Validation Instances: {n_validation}")
    
    ind_validation = random.sample(range(0, n_instances), n_validation)
    x_valid = x_all[ind_validation, :, :]
    y_valid = y_all[ind_validation]
    
    x_all = np.delete(x_all, ind_validation, axis = 0)
    y_all = np.delete(y_all, ind_validation, axis = 0)

    print(f"x_all: {x_all.shape} - y_all: {y_all.shape}")
    print(f"x_valid: {x_valid.shape} - y_valid: {y_valid.shape}")
    
    return {'x_all' : x_all, 'y_all' : y_all,
           'x_valid' :  x_valid, 'y_valid' : y_valid}