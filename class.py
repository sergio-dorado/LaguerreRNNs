import sys
from utils import *
from architectures_v1 import *
from datasets import *

if __name__ == "__main__":

    # Reading model and dataset
    model_name = sys.argv[1]
    dataset = sys.argv[2]

    # Validating model
    if not sys.argv[1] in LIST_OF_ARCHITECTURES:
        raise ValueError("Model not defined")

    # Validating dataset
    if not sys.argv[2] in LIST_OF_DATASETS:
        raise ValueError("Dataset not defined")

    mode = sys.argv[3]
    if mode == 'normal':
        k_fold = False
    else:
        k_fold = True

    # Verbosity
    if not sys.argv[4] is None:
        verbose = False
    else:
        verbose = True

    print("Model and dataset defined\nProceeding with experimentation")

    ################################
    ###### LOADING DATASETS ########
    ################################

    # Loading datasets
    data = train_test_valid_split(dataset)

    # Extracting datasets
    x_all, y_all, x_valid, y_valid = data['x_all'], data['y_all'], data['x_valid'], data['y_valid']

    if dataset == 'psMNIST':
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    else:
        x_train, y_train = None, None
        x_test, y_test = None, None

    if verbose: print(f"x_all: {x_all.valid} - y_all: {y_all.valid}\nx_valid: {x_valid.shape} - y_valid: {y_valid.shape}\n")

    #######################################
    ###### CREATING MODEL INSTANCE ########
    #######################################
    if verbose: print("Creating classifier\n")

    # Validating dataset dimensions
    if len(x_all.shape) == 2:
        # input_shape = (length, n_features)
        input_shape = (x_all.shape[1], 1)
    else:
        # input_shape = (length, n_features)
        input_shape = (x_all.shape[1], x_all.shape[-1])

    # Number of classes
    n_classes = to_categorical(y_all).shape[-1]

    # Classifier creation
    if model_name == "BRC":
        classifier = BRC_Classifier(input_shape = input_shape, n_classes = n_classes, dataset = dataset,
                                    output_dim = 400)
    elif model_name == "LMU":
        classifier = LMU_Classifier(input_shape = input_shape, n_classes = n_classes, dataset = dataset,
                                    units = 212, order = 256)
    elif model_name == "nBRC":
        classifier = nBRC_Classifier(input_shape = input_shape, n_classes = n_classes, dataset = dataset,
                                    output_dim = 400)
    elif model_name == "Laguerre":
        classifier = Laguerre_Classifier(input_shape = input_shape, n_classes = n_classes, dataset = dataset,
                                        units = 212, order = 256, variant = 'ct_laguerre', dt = 1)
    elif model_name == "Ladder":
        classifier = Ladder_Classifier(input_shape = input_shape, n_classes = n_classes, dataset = dataset,
                                      units = 212, max_delay = x_all.shape[1])
    elif model_name == "LSTM":
        classifier = LSTM_Classifier(input_shape = input_shape, n_classes = n_classes, dataset = dataset,
                                      units = 212)
    elif model_name == "GRU":
        classifier = GRU_Classifier(input_shape = input_shape, n_classes = n_classes, dataset = dataset,
                                      units = 212)

    ###################################
    ###### TRAINING CLASSIFIER ########
    ###################################
    classifier.fit(x_all, y_all, x_valid, y_valid,
                   batch_size = 100, verbose = 1, patience = 50, n_epochs = 2000, k_fold = k_fold, n_splits = 5,
                   x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test)
