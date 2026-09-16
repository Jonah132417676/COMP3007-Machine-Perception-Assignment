"""

task2.py

Given a cropped image, you must segment into individual digits (BPM) or segment area on thermometer with fluid mark.


Author: Zhong Cheng Lau 

Last Modified: 2026-16-09

"""

import os
import cv2
import glob

OUTPUT_DIR = os.path.join("output","task2")

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



def segment_lcd(image, lcd_name):
    """
    Segments the image into individual digits.
    
    """
    sub_dir = os.path.join(OUTPUT_DIR, lcd_name)

    h, w = image.shape[:2]

    num_digits = 7 # typical bpm reading is 3 (SYS) 2 (DIA) and 2 (PULSE)
    strip_w = w // num_digits # evenly separate the image into 7 segments width

    for d in range(1, num_digits + 1):
        # define x star and x end

        # THIS LOGIC IS NOT SUPPOSED TO BE CORRECT
        x_start = (d - 1) * strip_w
        x_end = d * strip_w if d < num_digits else w

        digit_img = image[:, x_start:x_end]

        save_output(os.path.join(sub_dir, f"d{d}.png"), digit_img, output_type='image')
    print(f"    [Task 2] {num_digits} digit images: saved t.png to {sub_dir}")

    
def segment_thermo(image, thermo_name):
    """
    Segments the thermo for fluid and markings. .
    
    """
    sub_dir = os.path.join(OUTPUT_DIR, thermo_name)

    h, w = image.shape[:2]
    # THIS LOGIC IS NOT SUPPOSED TO BE CORRECT
    crop = image[h // 3: 2 * h //3, :]

    save_output(os.path.join(sub_dir, "t.png"), crop, output_type='image')

    print(f"    [Task 2] {thermo_name}: saved t.png to {sub_dir}")
    
    
def run_task2(image_path, config):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # find all the .png images in the input directory
    png_files = sorted(glob.glob(os.path.join(image_path, "*.png")))

    if not png_files:
        print(" [Task 2] No input images found.")
        return

    print(f"    [Task 2] Found {len(png_files)} input images(s).")

    process_task2(png_files)

def process_task2(png_files):

    for file_path in png_files:
        fname = os.path.basename(file_path)
        base_name = os.path.splitext(fname)[0] # lcd2 or thermo1

        img = cv2.imread(file_path)
        if img is None or img.size == 0:
            print(f"    [Task 2] Could not read {fname}. Skipping")
            continue

        if fname.startswith("lcd"):
            segment_lcd(img, base_name)
        elif fname.startswith("thermo"):
            segment_thermo(img, base_name)
        else:
            print(f"    [Task 2] unknown file type: {fname}. Skipping")