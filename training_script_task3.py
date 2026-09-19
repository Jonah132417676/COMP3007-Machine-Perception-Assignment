"""

training_script_task3.py

Trains task3_digit_bpm_classifier and task3_thermometer_readings_classifier using YOLO neural network.


Author: Zhong Cheng Lau 

Last Modified: 2026-16-09

"""
import os
from ultralytics import YOLO

from ml_utils import train_YOLO, read_config_txt

LCD_TRAIN_DATA_PATH = os.path.join("training_data/task3/Task 3 Dataset LCD Digits/","data.yaml")
LCD_DIGIT_MODEL_NAME = "digit_LCD_classifier_model"

def train_model_BPM_reading(configs: dict) -> YOLO:
    """
    Train the BPM reading model digits.

    Input:
        - configs
    Output:
        - BPM classifier model 
    """
    # sample data  
    model, result = train_YOLO(LCD_DIGIT_MODEL_NAME, "data/task3/", LCD_TRAIN_DATA_PATH, configs)

    return model

def train_model_thermometer_reading(configs:dict) -> YOLO:
    """
    Train the thermometer reading model.

    Input:
        - configs
    Output:
        - model in "models/task3/task3/lcdX"
    """
    print()


if __name__ == "__main__":
    configs = read_config_txt()
    train_model_BPM_reading(configs)


# for digit reading uses shape context descriptors? for object detection.