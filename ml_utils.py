"""

ml_utils.py

A machine learning utils script that stores useful helper functions.

Functions:
    - 


"""
import os
import cv2
import numpy as np

from ultralytics import YOLO

def read_config_txt():
    """
    Wrapper for reading config base.
    """

    return read_config("config.txt")

# read the config file 
def read_config(config_path):
    """
    Reads the config file and outputs dictionary with keywords.
    """
    config = {}
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return config

    with open(config_path, 'r') as file:
        for line in file:
            if not line or ':' not in line: # skip non colon lines
                continue

            if line.strip() and not line.startswith('#'):
                key, value = line.split(':')
                config[key.strip()] = value.strip()
    return config

def read_image_file(image_path: str):
    """
    Reads the image file using image path.

    Input:
        - image_path
    Output:
        - image
    """
    # if path exists
    if os.path.exists(image_path):
        try: 
            img = cv2.imread(image_path)

            if img is not None:
                print(f"Successfully read image: {image_path}")

                return img
            else:
                print("Unable to read image: ", {e})

        except Exception as e:
            print(f"Cannot open image file, error: {e}")
    else:
        print("File does not exist: ", image_path)

# functions from practical

## TESTING ##

def test_KNN(knn: cv2.Algorithm, test_data: np.array, test_labels: np.array):
    """
    Tests model performance with kNN model and test data.

    Input:
        - knn -- knn model
        - test_data -- unseen data used to test the model after training
        - test_labels -- corresponding labels for the test data
    
    Output:
        - None
    """

    # test it with test data
    ret, result, neighbours, distance = knn.findNearest(test_data, k=k)

    accuracy = np.mean(result == test_labels) * 100
    error_rate = 100 - accuracy

    print(f"Error Rate for KNN: {error_rate}")

    # confusion matrix
    confusion_matrix = confusion_matrix(test_labels, result)
    print("Confusion Matrix: " + confusion_matrix)

def test_SVM(svm: cv2.Algorithm, test_data: np.array, test_labels:np.array):
    """
    Tests model performance with SVM model and test data.

    Input:
        - svm -- svm model
        - test_data -- unseen data used to test the model after training
        - test_labels -- corresponding labels for the test data
    
    Output:
        - None
    """

    test_result = svm.predict(test_data)[1].astype(np.int32).flatten()
    test_accuracy = np.mean(test_result == test_labels).flatten() * 100
    print(f"Test Accuracy: " + test_accuracy)

    # confusion matrix
    confusion_matrix = confusion_matrix(test_labels, test_result)
    print("Confusion Matrix: " + confusion_matrix)


## LOADING MODEL ##
def load_YOLO(path: str):
    """
    Loads a trained YOLO model from a give .pt file path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model weights not found at: '{path}'. "
            f"Ensure the path in your config is correct or that training has finished."
        )
    
    return YOLO(path)

## TRAINING ##
def train_YOLO(output_model_name: str, output_path: str, trainingDataPath: str, configs: dict):
    """
    Trains a model using YOLO (deep learning for object detection)
    """
    # load a model pretrained
    model = YOLO(configs["model"])

    #train on digits dataset
    results = model.train(
        data=trainingDataPath,
        epochs=int(configs["num_epochs"]),
        batch=int(configs["batch_size"]),
        lr0=float(configs["learning_rate"]),
        project=output_path,
        name=output_model_name,
        # DATA AUGMENTATION
        degrees=float(configs["degrees"]),
        translate=float(configs["translate"]),
        scale=float(configs["scale"]),
        save=True
    )

    print("TRAINED YOLO MODEL")
    return model

def train_KNN(output_model_name: str, output_path: str, k: int, train_data: np.array, train_labels: np.array, test_data: np.array, test_labels: np.array):
    """
    Trains a model using KNN (without preprocessing) and saves it to models output files.
    - NOTE: training data must be row samples layout.
    
    Input:
        - output_model_name -- name of output model
        - output_path -- name of output path to be saved
        - k -- k value for k nearest neighbours (amount of neighbours to check)
        - train_data -- (X) the training data for the model to train with
        - train_labels -- (y) the labels for the corresponding training data

    Output:
        - knn model
    """
    # create knn model
    knn = cv2.ml.KNearest_create()

    # train knn based on train data and labels with each row based on an sample
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    output_final_path = output_path + output_model_name + ".yml"
    knn.save(output_final_path)

    print("KNN MODEL SAVED")

    return knn


def train_SVM(output_model_name:str, output_path: str, linearSVM: bool, C: float, train_data: np.array, train_labels: np.array):
    """
    It is recommended to normalize training and test data (0 - 1).
    NOTE: ROW SAMPLES FOR DATA IS ASSUMED.

    Inputs:
        - linearSVM: boolean to decide if the svm boundary is linear or have more complex descision boundaries
        - C: value to determine margin influence in svm (C is higher more complex boundaries, C is less more generalized bounrday)
        - output_model_name -- name of output model
        - output_path -- name of output path to be saved
        - train_data -- (X) the training data for the model to train with
        - train_labels -- (y) the labels for the corresponding training data
        - test_data -- unseen data used to test the model after training
        - test_labels -- corresponding labels for the test data
    
    Output:
        - svm model
    """

    #svm
    svm = cv2.ml.SVM_create()

    # set parameters for svm
    if linearSVM:
        svm.setKernel(cv2.ml.SVM_LINEAR) # linear kernel
    else:
        # non linear kernel
        svm.setKernel(cv2.ml.SVM_RBF) #  non linear kernel
        svm.setGamma(0.001) # gamma (hyperparameter associated with kernel)

    svm.setC(C) # C
    svm.setType(cv2.ml.SVM_C_SVC) # use support vector classification

    # settings (convergence termination if too small)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TermCriteria_EPS, 100, 1E-8))

    # training
    svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    # predict training and test data
    output_final_path = output_path + output_model_name + ".yml"
    svm.save(output_final_path)

    print("SVM MODEL SAVED")
    return svm