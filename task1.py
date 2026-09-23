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
4. Use sift for keypoint descriptors between a template "bpm" or "thermometer"
5. Warp the image
6. Save the img.

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

ALLOW_BPM_DEBUG = True
ALLOW_THERMO_DEBUG = True

THERMO_LABEL = "thermometer"

BPM_LABEL = "blood pressure monitor"
BPM_THERMOMETER_DEETECTOR_MODEL_PATH = os.path.join("data/task1/","bpm_thermomter_detector.pt")
LCD_DISPLAY_DETECTOR_MODEL_PATH = os.path.join("data/task1/","lcd_display_detector.pt")

TEMPLATE_LCD_PATH = os.path.join("data/task1/templates/", "template_lcd.png")
TEMPLATE_THERMO_PATH = os.path.join("data/task1/templates/", "template_thermo.png")

BASE_CONFIDENCE_THRESHOLD_LCD_DISPLAY = 0.25
BASE_CONFIDENCE_THRESHOLD_BPM_THEROMETER = 0.25

# keypoint matching how many matches required
MIN_MATCH_COUNT = 10

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

    process_task1(images, ALLOW_BPM_DEBUG, ALLOW_THERMO_DEBUG)
   
def process_task1(images, allowBPMDebug, allowTHERMODebug):
    """
    
    Pipeline: Input -> [BPM THERMO Detector] -> 
    
    -> BPM -> [LCD Display Detector] -> [Crop Display] -> [Use orientated image for keypoint detection of four corners] -> [SIFT keypoint detection] -> [Perspective Transform] -> Output
    -> Thermo -> [Crop Thermometer] -> [Use orientated image SIFT for keypoint detection] -> [Perspective transform] -> Output
    
    """

    # load models
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

            # for each image if negative 
            print(f"LABEL: {label_name}, CONF: {box.conf[0].item()}")
            if label_name == THERMO_LABEL:
                final_image = process_thermometer_image(img, tensorBox, allowTHERMODebug)
                out_name = f"thermo{img_num}.png"
                print(f"    [Task 1] {basename}.jpg -> Thermometer -> {out_name}")
            elif label_name == BPM_LABEL:
                # pipeline: image -> [feature detection] -> [bpm detector] -> [lcd detector] ->  output image
                final_image = process_bpm_image(img, tensorBox, lcd_display_detector, allowBPMDebug)

                out_name = f"lcd{img_num}.png"

                print(f"    [Task 1] {basename}.jpg -> BPM -> {out_name}")
            else:
                print(f"    [Task 1] Object Detection not available for object label: {label_name}")
                out_name = ""
            
            output_path = os.path.join(OUTPUT_DIR, out_name)
            save_output(output_path, final_image, output_type='image')

def process_thermometer_image(img, tensorBox, allowDebug):
    """
    For a thermometer image. Map the thermometer to the template orientated lcd screen through SIFT.

    Input:
        - img -- image to warp
        - tensorBox -- box data on prediction of bpm
    
    Output:
        - final_image -- the final result image
    
    """
    # thermometer
    # obtain width and height of bounding box
    
    templateTHERMOImage = cv2.imread(TEMPLATE_THERMO_PATH)

    # crop it to the bounding boxes of thermometer
    final_image = crop_object_box(tensorBox, img)

    final_image = sift_keypoint_perspective_warp(templateTHERMOImage, final_image, allowDebug)

    if allowDebug:
        plt.imshow(final_image)
        plt.show()

    return final_image

def process_bpm_image(img, tensorBox, lcd_display_detector, allowDebug):
    """
    For a bpm image, detect the lcd and crop it. Then map the cropped lcd to the template orientated lcd screen through SIFT.

    Input:
        - img -- image to warp
        - tensorBox -- box data on prediction of bpm
    
    Output:
        - final_image -- the final result image
    
    """
    # read lcd template
    templateLCDImage = cv2.imread(TEMPLATE_LCD_PATH)

    # bpm (crop to bounding boxes)
    bpmImg = crop_object_box(tensorBox, img)

    # detect lcd screen using lcd screen detector
    lcdDisplayBox = lcd_display_detector.predict(bpmImg, conf=BASE_CONFIDENCE_THRESHOLD_LCD_DISPLAY, retina_masks=True)[0].boxes.xyxy[0]
    # crop it (to lcd screen)
    lcdCropped = crop_object_box(lcdDisplayBox, bpmImg)

    final_image = sift_keypoint_perspective_warp(templateLCDImage, lcdCropped, allowDebug)
    if allowDebug:
        plt.imshow(final_image)
        plt.show()

    return final_image

def sift_keypoint_perspective_warp(image1, image2, allowDebug):
    """
    Maps the keypoints of image2 to the dimensions of image1.

    Input:
        - image1 -- an image to match (template)
        - image2 -- an image to match (warped)

    Output:
        - final_image -- the final warpped image
    """

    goodMatches, kp1, kp2 = obtain_good_matches(image1, image2)
    
    # if good enough matches
    if len(goodMatches) > MIN_MATCH_COUNT:
        # obtain matching keypoints locations of both images
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)

        # map from image 2 to image 1's frame work
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # apply perspective transform to find matching borders
        height, width = image1.shape[:2]

        if allowDebug:
            draw_params = dict(matchColor = (0, 255, 0),
                               singlePointColor=None,
                               matchesMask=matchesMask, # draw only inliers
                               flags=2)
            debugImg = cv2.drawMatches(image1, kp1, image2, kp2, goodMatches, None, **draw_params)
            plt.imshow(debugImg)
            plt.show()
      
        # find top Left, 
        final_image = cv2.warpPerspective(image2, M, (width, height))
        return final_image
    else:
        print(f"    [Task 1] Not enough matches are found, current: {len(goodMatches)}, needed: {MIN_MATCH_COUNT}")
        return None
    
def obtain_good_matches(image1, image2):
    """
    Find the best matches between two images. 
    
    Input:
        - image1 -- an image to match
        - image2 -- an image to match
    
    Output:
        - good - the good matches found through Lowe's ratio test
        - kp1 - key point locations on image 1
        - kp2 - key point locations on image 2
    
    
    """
    # use sift to compare orientation and their corresponding points
    # https://docs.opencv.org/4.13.0/d1/de0/tutorial_py_feature_homography.html
    # initialise SIFT
    sift = cv2.SIFT_create()
    # find the keypoints descriptors of both image
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    index_params = dict(algorithm=1, trees= 5)
    search_params = dict(checks = 50)

    # find keypoints
    flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
    matches = flann.knnMatch(des1, des2, k = 2)

    # store the good matches using Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return good, kp1, kp2