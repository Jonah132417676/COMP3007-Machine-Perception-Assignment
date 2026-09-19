"""

task2.py

Given a cropped image, you must segment into individual digits (BPM) or segment area on thermometer with fluid mark.


Author: Zhong Cheng Lau 

Last Modified: 2026-16-09

"""

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from ml_utils import load_YOLO

OUTPUT_DIR = os.path.join("output","task2")
LCD_MODEL_PATH = os.path.join("data/task3/digit_LCD_classifier_model/","lcd_digit_detector.pt")

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

    # First run the lcd digit detector into the image
    digitDetector = load_YOLO(LCD_MODEL_PATH)

    result = digitDetector.predict(image, retina_masks = True)[0]
    

    # sort by y then x
    digitBoxesOrder = [box for box in result.boxes]
    # sort by y -> box.xywh[0].tolist()[1] = y center
    digitBoxesOrder.sort(key=lambda box: box.xywh[0].tolist()[1])

    # 0-2 is sys, 3-4 dia, 5-6 pulse --> SEPARATE
    sysBoxes, diaBoxes, pulseBoxes = digitBoxesOrder[:3], digitBoxesOrder[3:5], digitBoxesOrder[5:7]

    # sort by x -> box.xywhh[0].tolist()[0] = x center
    sysBoxes.sort(key=lambda box: box.xywh[0].tolist()[0])
    diaBoxes.sort(key=lambda box: box.xywh[0].tolist()[0])
    pulseBoxes.sort(key=lambda box: box.xywh[0].tolist()[0])

    # then print 
    
    plot_lcd_digits(result, sysBoxes, diaBoxes, pulseBoxes)

    # total ordered boxes, sys, dia then pulse
    totalOrderedBoxes = [*sysBoxes, *diaBoxes, *pulseBoxes]

    num_digits = len(totalOrderedBoxes)
    class_names = result.names
    for i, box in enumerate(totalOrderedBoxes):
        # define x star and x end
        class_id = int(box.cls[0].item())
        # obtain label name
        d = class_names[class_id] 

        x1, y1, x2, y2 = box.xyxy[0]

        digit_img = image[int(y1):int(y2), int(x1):int(x2)]

        save_output(os.path.join(sub_dir, f"d{i+1}.png"), digit_img, output_type='image')
        print(f"    [Task 2] {num_digits} digit images: saved t.png to {sub_dir}")
    
def segment_thermo(image, thermo_name):
    """
    Segments the thermo for fluid and markings.
    
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


def plot_lcd_digits(result, sysBoxes, diaBoxes, pulseBoxes):
    plt.imshow(result.plot())
        
    plot_points(sysBoxes, result, "red")
    plot_points(diaBoxes, result, "blue")
    plot_points(pulseBoxes, result, "green")

    plt.show()

def plot_points(boxes, result, colour = "red"):
    class_names = result.names
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0].item())
        # obtain label name
        label_name = class_names[class_id] 
        print(f"ORDER: {label_name}")

        # obtain position
        xcenter, ycenter, width, height = box.xywh[0].tolist()
        plt.scatter(xcenter, ycenter, color=colour, s=40, zorder=5)