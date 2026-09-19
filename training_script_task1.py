"""

training_script_task1.py 

Trains a model that has object detection capabilities for BPM lcd display or thermometer class for task 1.


Author: Zhong Cheng Lau 

Last Modified: 2026-16-09

"""

import os 
import sys
from ultralytics import YOLO

from ml_utils import train_YOLO, read_config_txt

BPM_THERMOMETER_TRAIN_DATA_PATH = os.path.join("training_data/task1/BPM and Thermometer/","data.yaml")
LCD_DISPLAY_DETECTOR_DATA_PATH = os.path.join("training_data/task1/LCD Display Detector/","data.yaml")

def train_model_bpm_thermometer(configs: dict) -> YOLO:
    """
    Train the BPM reading model digits.

    Input:
        - configs
    Output:
        - BPM classifier model 
    """
    # sample data for roboflow
  
    model, results = train_YOLO("task1_model", "data/task1/", BPM_THERMOMETER_TRAIN_DATA_PATH, configs)

    return model

def train_lcd_display_detector(configs: dict) -> YOLO:
    """
    Train the BPM reading model digits.

    Input:
        - configs
    Output:
        - BPM classifier model 
    """
    # sample data for roboflow
  
    model, results = train_YOLO("lcd_detector", "data/task1/", LCD_DISPLAY_DETECTOR_DATA_PATH, configs)

    return model


def printout_message():
    """
    Prints the usage notes for this script.

    """

    print("USAGE: training_script_task1.py <task number>")
    print("""
    Example:
        python3 assignment.py 1

        1 - train model bpm thermometer
        2 - train lcd detector
""")

if __name__ == "__main__":
    printout_message()

    configs = read_config_txt()
    if len(sys.argv) == 2:
        if sys.argv[1] == '1':
            train_model_bpm_thermometer(configs)
        elif sys.argv[1] == '2':
            train_lcd_display_detector(configs)
        else:
            print("wrong system arguments")
            printout_message()