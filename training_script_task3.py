"""

training_script_task3.py

Trains task3_digit_bpm_classifier and task3_thermometer_readings_classifier using YOLO neural network.


Author: Zhong Cheng Lau 

Last Modified: 2026-09-09

"""

from ultralytics import YOLO

from ml_utils import train_YOLO, read_config_txt


def train_model_BPM_reading(configs: dict) -> YOLO:
    """
    Train the BPM reading model digits.

    Input:
        - configs
    Output:
        - BPM classifier model 
    """
    # sample data
    bpmDigitFolderPath = "data/task3/LCD Digits.v1i.yolov8/data.yaml"
  
    model = train_YOLO("task3_digit_bpm_classifier", "models/task3/", bpmDigitFolderPath, configs)

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