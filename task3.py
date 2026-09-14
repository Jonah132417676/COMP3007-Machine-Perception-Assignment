"""

task3.py

HOW TO USE:
    - cmd line: python3 task3.py image_path


Task 3: Reading recognition and calculation.

A module that will be given a path to an input directory containing sub directories, one per region.
Each sub directory is named either lcdX for BPM, or thermoX/ for thermometer. 
BPM subdirectories contain individual digit images.
Thermometer sub directories contain a single segmented region image t.png

1. For each BPM sub directory, write a Python program that reads each digit image and outputs the recognised digit to a text file. Use a classification model for digit recognition that is either trained or fine-tuned yourself.
Output:
    - A text file dZ.txt containing only the recognised digit.

2.For each thermometer subdirectory, write a python programthat detects the fluid
endpoint position (red colour) positions from t.png, then calculate the current temperature reading.
- The program should interpolate between nearest scale markings to estimate temperature.
Output:
    - A text file t.txt containing a single interger representing the estimated temperature in degrees celcius.

    
Author: Zhong Cheng Lau 

Last Modified: 2026-09-09

"""
import sys
import cv2
import os
import math

from ultralytics import YOLO

from ml_utils import read_image_file, train_YOLO, load_YOLO, read_config_txt

"""
Test Path Files:
validation/task3/lcd2/d1.png

"""

def run_task3(image_path: str, config: dict):
    BPM_reading_detection(image_path)

def BPM_reading_detection(image_path: str, configs:dict):
    """
    Classifying number and output to a dZ.txt file under output/task3/lcdX/.

    Framework: 
        openCV - support vector machines

    Input:
        - image_path

    Output: (txt file in output task3)
        - None 
    """
    output_path = "output/task3/"
    output_name = "dx.txt"
    output_final_path = output_path + output_name

    model_path = "runs/detect/models/task3/task3_digit_bpm_classifier/weights/best.pt"
    model = load_YOLO(model_path)

    results = model.predict(source=image_path, conf=float(configs["confidence_threshold"]))

    # predicted digit has the highest confidence
    predicted_digit = None # get predicted digit from idx
    currentConfidence = -1 * float("inf")  

    for result in results:
        boxes = result.boxes
        result.show()
        print(f"Amount of Boxes: {len(boxes)}")
        for box in boxes:
            class_id = int(box.cls[0])
            # get class label name
            label = model.names[class_id]
            # get bounding box coordinate
            coords = box.xyxy[0].tolist()
            # get confidence score
            conf = float(box.conf[0])

            print(f"\nLabel: {label}, Coords: {coords}, Conf: {conf}")

            if conf > currentConfidence:
                currentConfidence = conf
                predicted_digit = label

    # save to file
    if predicted_digit:
        with open(output_final_path, "w") as fo:
            fo.write(predicted_digit)
            print(predicted_digit)

    else:
        print(f"Unable to find predicted digit for the image. No. Results: {len(results)}")





def main():

    print("""
########### TASK 3 MENU ############
Get result test image input.

python3 task3.py <image_path>

""")
    if len(sys.argv) == 2:
        try:
            image_path = sys.argv[1]

            BPM_reading_detection(image_path, read_config_txt())   
        except Exception as e:
            print(f"Failed to perform task3 task error: {e}")
    else:
        print("Did not provide sufficient arguments.")


if __name__ == "__main__":
    main()
    