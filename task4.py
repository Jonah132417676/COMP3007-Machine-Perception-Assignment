"""

task4.py

Runs the entire pipeline from start to finish.

Author: Zhong Cheng Lau 

Last Modified: 2026-24-09

"""

import os
import cv2
import glob
import random

OUTPUT_DIR = os.path.join("output","task4")
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

def run_task4(image_path, config):

    # find all the jpg images in input directory
    images = sorted(glob.glob(os.path.join(image_path, "img*.jpg")))

    if not images:
        print("     [Task 4] No input images found.")
        return

    print(f"    [Task 4] Found {len(images)} input image(s).")

    process_task4(images)


def get_bpm_reading():
    sys_val = -90
    dia_val = -50
    pulse = -40
    return f"{sys_val}, {dia_val}, {pulse}"

def get_thermo_reading():
    return str(-999)

def process_task4(images):
  
    for i, img_file in enumerate(images):
        basename = os.path.splitext(os.path.basename(img_file))[0]

        # identification [TASK 1] and crop


      
        if i == "thermometer":
            # themoemeter
            reading = get_thermo_reading()
            line = f"thermo {reading}"
            print(f"    [Task 4] {basename}.jpg -> {line}")
        elif i == "bpm":
            # BPM
            reading = get_bpm_reading()
            line = f"bpm {reading}"
            print(f"    [Task 4] {basename}.jpg -> {line}")
        else:
            # negative produces no output
            print(f"    [Task 4] {basename}.jpg -> NEGATIVE (no output)")
            continue

        save_output(os.path.join(OUTPUT_DIR, f"{basename}.txt"), line, output_type = "txt")