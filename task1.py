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

Last Modified: 2026-16-09

"""

import os
import cv2
import glob
import matplotlib.pyplot as plt

from ml_utils import load_YOLO
THERMO_LABEL = "thermometer"
BPM_LABEL = "blood pressure monitor"
MODEL_PATH = os.path.join("models/task1/task1_model/weights/","best.pt")
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

    yolo = load_YOLO(MODEL_PATH)

     # task 1 logic
    for i, img_file in enumerate(images):
        img = cv2.imread(img_file)
        if img is None:
            print(f"    [Task 1] Could not read {img_file}. Skipping.")
            continue

        # extract image number from the file name (img2.jpg -> 2)
        basename = os.path.splitext(os.path.basename(img_file))[0]
        img_num = basename.replace("img", "")

        results = yolo.predict(img)
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

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            # obtain label name
            label_name = class_names[class_id]
            # for each image if negative 
            print(f"LABEL: {label_name}, CONF: {box.conf[0].item()}")
            if label_name == THERMO_LABEL:
                # thermometer

                # crop it

                # orientate the image
                final_image = img


                out_name = f"thermo{img_num}.png"
                cv2.imshow("thermometer", final_image)
                print(f"    [Task 1] {basename}.jpg -> Thermometer -> {out_name}")
            elif label_name == BPM_LABEL:
                # bpm

                # crop it (to lcd screen)

                # orientate


                final_image = img
                out_name = f"lcd{img_num}.png"
                cv2.imshow("bpm", final_image)

                print(f"    [Task 1] {basename}.jpg -> BPM -> {out_name}")
            else:
                print(f"    [Task 1] Object Detection not available for object label: {label_name}")
                out_name = ""
            
            output_path = os.path.join(OUTPUT_DIR, out_name)
            save_output(output_path, final_image, output_type='image')

    