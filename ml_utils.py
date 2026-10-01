"""

ml_utils.py

A machine learning utils script that stores useful helper functions.

Author: Zhong Cheng Lau

Last Modified: 2026-1-10


"""
import os
import shutil
import cv2
import numpy as np
import wandb
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
from pathlib import Path

def crop_object_box(box, img):
    """
    Crops the image by the bounding boxes.

    Inputs:
        - box -- bounding box data
        - img -- image to crop
    
    Output:
        - cropped_image 
    
    """

    x1, y1, x2, y2 = map(int, box[:4])
    return img[y1:y2, x1:x2]

def perspective_transform_image(img, pts_src, pts_dst, width, height):
    """
    Perform perspective transform on the image.

    Inputs:
        - img -- the image to apply perspective transform to
        - pts_src -- source points for the image
        - pts_dst -- subsequent destination points for the perspective transform
        - width -- width of the image
        - height -- height of the image 

    
    """
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warpedImg = cv2.warpPerspective(img, matrix, (width, height))
    return warpedImg

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


## LOADING MODEL ##
def load_YOLO(path: str):
    """
    Loads a trained YOLO model from a give .pt file path.
    """
    final_model_path = str(Path(path).resolve())

    if not os.path.exists(final_model_path):
        raise FileNotFoundError(
            f"Model weights not found at: '{final_model_path}'. "
            f"Ensure the path in your config is correct or that training has finished."
        )
    
    return YOLO(final_model_path)

## TRAINING ##
def train_YOLO(output_model_name: str, output_path: str, trainingDataPath: str, configs: dict):
    """
    Trains a model using YOLO (deep learning for object detection)
    NOTE: REQUIRES GPU

    Inputs:
        - output_model_name -- name of the model output 
        - output_path -- output path of the model
        - trainingDataPath -- path of the training data contaiing data.yaml
        - configs -- dictionary including training parameters
    
    """


    print("GPU Available: ", torch.cuda.is_available())


    # initialise weights and biases run
    wandb.init(
        entity="jonahzclau-curtin-university",    
        project=f"ultralytics-{output_model_name}", 
        job_type="training",
        config=configs
    )

    # load a model pretrained
    model = YOLO(configs["model"])

    # add callback function
    add_wandb_callback(model, enable_model_checkpointing=True) # set up checkpoints so we can go back
    
    final_training_data_path = str(Path(trainingDataPath).resolve())
    #train on digits dataset
    results = model.train(
        data=final_training_data_path,
        epochs=int(configs["num_epochs"]),
        batch=int(configs["batch_size"]),
        lr0=float(configs["learning_rate"]),
        name=output_model_name,        
        save=True,
        exist_ok = True,
        device="cuda"
    )

    # validation for the model
    model.val()

    # finish wandb run
    wandb.finish()

    print("TRAINED YOLO MODEL")
    return model, results


def hsv_threshold(image, lower = (5, 75, 55), upper = (170, 255, 255)):
    """
    Applies a hsv threshold to the image between lowwe NS UPPWE.

    Inputs:
        - image -- the image to apply effect to
        - lower -- lower bound for hue 
        - upper -- higher bound for hue
    
    Outputs:
        - cleaned_image -- the cleaned image for thresholding.
    """
    
    # convert to hsv (hue, sat, val) space img 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)

def clear_file_dir(folder_path):
    """
    Clears the file directory and its contents. https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder

    Inputs:
        - folder_path -- the path to the directory to clear
    """

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                # if file unlink
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                # rm tree if directory
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {folder_path}. Reason: {e}')

def validate_model_matrix(model, dataset_path, title):    
    """
    Validation set on model.

    Inputs:
        - model -- the model to validate
        - dataset_path -- path of the dataset to validate with
    
    """
    metrics = model.val(data=dataset_path)
    confusion_matrix = metrics.confusion_matrix.matrix

    class_names = list(model.names.values())

    # plt confusion matrix 
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data=confusion_matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()

