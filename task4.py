"""

task4.py

Runs the entire pipeline from start to finish.

Author: Zhong Cheng Lau 

Last Modified: 2026-25-09

"""

import os
import cv2
import glob

from task1 import run_task1
from task2 import run_task2
from task3 import run_task3

OUTPUT_DIR_TASK4 = os.path.join("output","task4")
OUTPUT_DIR_TASK1= os.path.join("output", "task1")
OUTPUT_DIR_TASK2= os.path.join("output", "task2")
OUTPUT_DIR_TASK3= os.path.join("output", "task3")

def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")


def get_bpm_reading(img_num):
    lcd_dir = os.path.join(OUTPUT_DIR_TASK3, f"lcd{img_num}")

    # structure:
    # 3 is sys (1, 2, 3)
    # 2 is dia (4, 5)
    # 2 is pulse (6, 7)

    sys_val = ""
    dia_val = ""
    pulse_val = ""
    for digitNum in range(1, 8):
        # read file
        file_path = os.path.join(lcd_dir, f"d{digitNum}.txt")
        if os.path.isfile(file_path):
            with open(file_path, 'r') as fo:
                digitStr = str(fo.readline().strip())
                if digitNum <= 3:
                    sys_val += digitStr
                elif digitNum <= 5:
                    dia_val += digitStr
                elif digitNum <= 7:
                    pulse_val += digitStr
                else:
                    print(f"    [Task 4] The digit num is not within 1-7 range for num: {digitNum}")
        else:
            print(f"     [Task 4] Did not find the lcd file: {file_path}")
            return f"     [Task 4] Did not find the lcd file: {file_path}"
        

   
    return f"{sys_val},{dia_val},{pulse_val}"

def get_thermo_reading(img_num):

    # read from output of task3 
    thermo_dir = os.path.join(OUTPUT_DIR_TASK3, f"thermo{img_num}")

    # read the t.txt file
    file_path = os.path.join(thermo_dir, 't.txt')
    if os.path.isfile(file_path):
        with open(file_path,'r') as fo:
            # read the first line
            line = fo.readline().strip()
            return str(line)
    else:
        print(f"     [Task 4] Did not find the thermo file: {file_path}")
        return f"Did not find the thermo file: {file_path}"

def process_task4(image_path, images, config):

    # task 1 -> object detection and orientation
    run_task1(image_path, config)

    # task 2 -> segmentation from output of task1
    run_task2(OUTPUT_DIR_TASK1, config)

    # task 3 
    run_task3(OUTPUT_DIR_TASK2, config)

    for i, img_file in enumerate(images):
        basename = os.path.splitext(os.path.basename(img_file))[0] # img1
        img_num = basename.replace("img", "") # 1

        # Check if task 3 created output directories for this specific image
        thermo_dir = os.path.join(OUTPUT_DIR_TASK3, f"thermo{img_num}")
        lcd_dir = os.path.join(OUTPUT_DIR_TASK3, f"lcd{img_num}")

        # thermometer reading exists
        if os.path.exists(thermo_dir) and os.path.isfile(os.path.join(thermo_dir, 't.txt')):
            # themoemeter
            reading = get_thermo_reading(img_num)
            line = f"thermo {reading}"
            print(f"    [Task 4] {basename}.jpg -> {line}")

        elif os.path.exists(lcd_dir):
            # BPM
            reading = get_bpm_reading(img_num)
            line = f"bpm {reading}"
            print(f"    [Task 4] {basename}.jpg -> {line}")
        else:
            # negative produces no output (no file exists)
            print(f"    [Task 4] {basename}.jpg -> NEGATIVE (no output)")
            continue

        save_output(os.path.join(OUTPUT_DIR_TASK4, f"{basename}.txt"), line, output_type = "txt")

def run_task4(image_path, config):

    # find all the jpg images in input directory
    images = sorted(glob.glob(os.path.join(image_path, "img*.jpg")))

    if not images:
        print("     [Task 4] No input images found.")
        return

    print(f"    [Task 4] Found {len(images)} input image(s).")
    
    process_task4(image_path, images, config)