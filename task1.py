"""

task1.py

python3 assignment.py task1 data/testing/validation/task1 config.txt


Detect if in the image has bpm, thermometer or nothing, then crop it and orientate it for its output. 
Object Detection: YOLO 
    - extract bounding boxes and labels from yolo model for task1
    - determine label
    - crop area
    - orientate it properly
    - save it

1. Determine if valid medical device is present.
2. Classify the device as either a blood presure monitor or a wall thermometer.
3. Extract the reading region.
4. Apply geometric correction.
5. Save the img.
BPM - lcd screen
thermometer - cropped therometer image
negative - no output


Author: Zhong Cheng Lau 

Last Modified: 2026-20-09

"""

import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import math

from ml_utils import load_YOLO, crop_object_box, perspective_transform_image
THERMO_LABEL = "thermometer"
BPM_LABEL = "blood pressure monitor"
BPM_THERMOMETER_DEETECTOR_MODEL_PATH = os.path.join("data/task1/","bpm_thermomter_detector.pt")
LCD_DISPLAY_DETECTOR_MODEL_PATH = os.path.join("data/task1/","lcd_display_detector.pt")

BASE_CONFIDENCE_THRESHOLD_LCD_DISPLAY = 0.25
BASE_CONFIDENCE_THRESHOLD_BPM_THEROMETER = 0.25

OUTPUT_DIR = os.path.join("output","task1")

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


def run_task1(image_path, config):
    """
    Entry point for task1 called by assignment.py

    Inputs:
        - image_path -- Path to the input directory for this task
        - config -- config file for parameters
    
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # find all the .jpg images in the input directory
    images = sorted(glob.glob(os.path.join(image_path, "img*.jpg")))

    if not images:
        print(" [Task 1] No input images found.")
        return

    print(f"    [Task 1] Found {len(images)} input images(s).")

    process_task1(images)
   
def process_task1(images):
    # load model ->

    bpm_model_detector = load_YOLO(BPM_THERMOMETER_DEETECTOR_MODEL_PATH)
    lcd_display_detector = load_YOLO(LCD_DISPLAY_DETECTOR_MODEL_PATH)
     # task 1 logic
    for i, img_file in enumerate(images):
        img = cv2.imread(img_file)
        if img is None:
            print(f"    [Task 1] Could not read {img_file}. Skipping.")
            continue

        # extract image number from the file name (img2.jpg -> 2)
        basename = os.path.splitext(os.path.basename(img_file))[0]
        img_num = basename.replace("img", "")

        results = bpm_model_detector.predict(img, conf=BASE_CONFIDENCE_THRESHOLD_BPM_THEROMETER, retina_masks=True)
        result = results[0]
        # for each result    
        class_names = result.names

        plt.imshow(result.plot())
        plt.show()


        # NEGATIVE
        if result is None or len(result.boxes) == 0:
            # negative produce no output
            final_image = None
            print(f"    [Task 1] {basename}.jpg -> NEGATIVE (no output)")
            continue

        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0].item())
            # obtain label name
            label_name = class_names[class_id]
            tensorBox = results[0].boxes.xyxy[i]
            x1, y1, x2, y2 = map(int, tensorBox[:4])

            # obtain width and height of bounding box
            width = abs(x2 - x1)
            height = abs(y2 - y1)



            # for each image if negative 
            print(f"LABEL: {label_name}, CONF: {box.conf[0].item()}")
            if label_name == THERMO_LABEL:
                # thermometer


                # crop it to the bounding boxes
                final_image = crop_object_box(tensorBox, img)
                # orientate the image (rotate by theta optimal)

                final_image = apply_perspective_transform(final_image, width, height)


                out_name = f"thermo{img_num}.png"
                print(f"    [Task 1] {basename}.jpg -> Thermometer -> {out_name}")
            elif label_name == BPM_LABEL:
                # pipeline: image -> [feature detection] -> [bpm detector] -> [lcd detector] ->  output image

                # bpm (crop to bounding boxes)
                final_image = crop_object_box(tensorBox, img)

                # detect lcd screen using lcd screen detector
                lcdDisplayBox = lcd_display_detector.predict(final_image, conf=BASE_CONFIDENCE_THRESHOLD_LCD_DISPLAY, retina_masks=True)[0].boxes.xyxy[0]
                # crop it (to lcd screen)
                final_image = crop_object_box(lcdDisplayBox, final_image)
                # orientate

                

                out_name = f"lcd{img_num}.png"

                print(f"    [Task 1] {basename}.jpg -> BPM -> {out_name}")
            else:
                print(f"    [Task 1] Object Detection not available for object label: {label_name}")
                out_name = ""
            
            output_path = os.path.join(OUTPUT_DIR, out_name)
            save_output(output_path, final_image, output_type='image')



def apply_perspective_transform(crop, width, height):
    # convert to gray scale
    img = crop.copy()
    pts_src = np.float32([[50, 100], [400, 50], [450, 500], [20, 450]])

    pts_dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    return perspective_transform_image(crop, pts_src, pts_dst, width, height)