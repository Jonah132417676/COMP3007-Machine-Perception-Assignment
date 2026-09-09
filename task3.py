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

from ml_utils import read_image_file

"""
Test Path Files:
validation/task3/lcd2/d1.png

"""


def run_task3(image_path: str, config: dict):
    print()

def BPM_reading_detection(image_path: str):
    """
    Classifying number and output to a dZ.txt file under output/task3/lcdX/.

    Framework: 
        openCV - support vector machines

    Input:
        - image_path

    Output: (txt file in output task3)
        - None 
    """
    output_path = "output/task3/lcdX/"

    image = read_image_file(image_path)
    print(image)

def train_model_BPM_reading():
    """
    Train the BPM reading model digits.

    Output:
        - model in "models/task3/"
    """





def main():
    # image path input argument passed
    if len(sys.argv) > 1: 
        image_path = sys.argv[1]

        BPM_reading_detection(image_path)
    else:
        print("No image file path given in argv.")

if __name__ == "__main__":
    main()