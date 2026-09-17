"""

training_script_task1.py 

Trains a model that has object detection capabilities for BPM lcd display or thermometer class for task 1.


Author: Zhong Cheng Lau 

Last Modified: 2026-16-09

"""

import os 
from ultralytics import YOLO

from ml_utils import train_YOLO, read_config_txt

TRAIN_DATA_PATH = os.path.join("training_data/task1/Task 1 Dataset/","data.yaml")

def train_model_bpm_thermometer(configs: dict) -> YOLO:
    """
    Train the BPM reading model digits.

    Input:
        - configs
    Output:
        - BPM classifier model 
    """
    # sample data for roboflow
  
    model, results = train_YOLO("task1_model", "models/task1/", TRAIN_DATA_PATH, configs)

    return model



if __name__ == "__main__":
    configs = read_config_txt()
    train_model_bpm_thermometer(configs)
