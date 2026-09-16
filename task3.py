"""

task3.py

Produce predictions on the digit readings or the thermometer readings.


Author: Zhong Cheng Lau 

Last Modified: 2026-16-09

"""

import os
import cv2
import glob

OUTPUT_DIR = os.path.join("output","task3")
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

def recognise_lcd_digits(sub_dir_path, lcd_name):
    """
    Recognise the digit from the image.
    
    """
    out_dir = os.path.join(OUTPUT_DIR, lcd_name)

    digit_files = sorted(glob.glob(os.path.join(sub_dir_path, "d*.png")))

    if not digit_files:
        print(f"    [Task 3] {lcd_name}: no digit images found.")

    for dfile in digit_files:
        dname = os.path.splitext(os.path.basename(dfile))[0] # d1, d2 ...

        # NOT CORRECTLY IMPLEMENTED
        digit = -1

        save_output(os.path.join(out_dir, f"{dname}.txt"), str(digit), output_type="txt")

    print(f"    [Task 3] {lcd_name}: recognised {len(digit_files)} digit(s) -> {out_dir}")

def calculate_temperature(sub_dir_path, thermo_name):

    out_dir = os.path.join(OUTPUT_DIR, thermo_name)

    # check that t.png exists
    t_path = os.path.join(sub_dir_path, "t.png")
    if not os.path.isfile(t_path):
        print(f"    [Task 3] {thermo_name} : t.png not found. Skipping.")
        return

    # NOT CORRECT
    temperature = -999

    save_output(os.path.join(out_dir, "t.txt"), str(temperature), output_type="txt")    
    print(f"    [Task 3] {thermo_name}: esimated temperature = {temperature}")

def run_task3(image_path, config):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # find all the .sub directories in the input directory
    if not os.path.isdir(image_path):
        print(f"    [Task 3] Input path does not exist: {image_path}")

    # obtain all sub directories
    sub_dirs = sorted([d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))])

    if not sub_dirs:
        print(f"    [Task 3] No sub directories found in input")

    print(f"    [Task 3] Found {len(sub_dirs)} sub-directoy(ies).")

    process_task3(sub_dirs, image_path)
   
def process_task3(sub_dirs, image_path):

    for dname in sub_dirs:
        sub_dir_path = os.path.join(image_path, dname)

        if dname.startswith("lcd"):
            recognise_lcd_digits(sub_dir_path, dname)
        elif dname.startswith("thermo"):
            calculate_temperature(sub_dir_path, dname)
        else:
            print(f"    [Task 3] Unknown sub-directory: {dname}. Skipping.")