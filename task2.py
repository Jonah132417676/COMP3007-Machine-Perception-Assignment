"""

task2.py

Given a cropped image, you must segment into individual digits (BPM) or segment area on thermometer with fluid mark.


Author: Zhong Cheng Lau 

Last Modified: 2026-20-09

"""

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from ml_utils import load_YOLO, hsv_threshold

ALLOW_BPM_DEBUG = True
ALLOW_THERMO_DEBUG = True

OUTPUT_DIR = os.path.join("output","task2")
LCD_MODEL_PATH = os.path.join("data/task3/","lcd_digit_detector.pt")

HUE_THRESHOLD_THERMOMETER = 70
HOUGH_LINES_THRESHOLD_FLUID = 70
THERMO_ZOOM_DIVIDER = 15
DILATE_KERNEL_SIZE = 5

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



def segment_lcd(image, lcd_name, allowDebug):
    """
    Segments the image into individual digits.

    Pipeline: Input -> [YOLOv8 detect all lcd digits] -> [Order Boxes by Y highest] -> [Separate into rows, Sys(3), Dia(2), Pulse(2)] -> [Order Boxes in each sys, dia and pulse by x order] -> [Recombine into ordered full list sys, dia, pulse] -> [Crop images bounding box] -> output
    
    Inputs:
        - image -- input lcd display image
        - lcd_name -- name of lcd image

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
    if allowDebug:
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
    
def segment_thermo(image, thermo_name, allowDebug):
    """
    Segments the thermo for fluid and markings.

    Pipeline: Input -> [HSV convert] -> [HSV thresholding pink/red] -> [Morphology Dilate (fluid wider)] -> [Gray Scale] -> [Canny Edge Detector] -> [Hough Lines Detector] -> [Filter Valid Lines] -> [Find minimum y (highest) line point] -> [Crop height at that point] -> output
    
    Get rid of outlier lines from non main line

    Input:
        - image -- input thermometer image cropped
        - thermo_name -- the name of the thermometer  png
    """
    sub_dir = os.path.join(OUTPUT_DIR, thermo_name)

    h, w = image.shape[:2]

    x, y_maximum_point = get_highest_fluid_endpoint(image,  DILATE_KERNEL_SIZE, allowDebug)
    
    zoomHeight = h // THERMO_ZOOM_DIVIDER # height of crop will be a quarter of total thermometer height
    crop = image[y_maximum_point - zoomHeight: y_maximum_point + zoomHeight, :]
    if allowDebug:
        plt.imshow(crop)
        plt.show()

    print(f"    [Task 2] {thermo_name}: saved t.png to {sub_dir}")

    save_output(os.path.join(sub_dir, "t.png"), crop, output_type='image')


def plot_lcd_digits(result, sysBoxes, diaBoxes, pulseBoxes):
    """
    Shows lcd digits and their labels.

    Input:
        - result -- the box results from predictions
        - sysBoxes -- ordered boxes of SYS on bpm
        - diaBoxes -- ordered boxes of DIA on bpm
        - pulseBoxes -- ordered boxes of PULSE on bpm

    
    """
    plt.imshow(result.plot())
        
    plot_points(sysBoxes, result, "red")
    plot_points(diaBoxes, result, "blue")
    plot_points(pulseBoxes, result, "green")

    plt.show()

def plot_points(boxes, result, colour = "red"):
    """
    Plots the points of the center of the boxes to the colour.

    Inputs:
        - boxes -- the boxes to paint
        - result -- the result predictions bounding boxes
        - colour -- the colour to paint
    """
    class_names = result.names
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0].item())
        # obtain label name
        label_name = class_names[class_id] 
        print(f"ORDER: {label_name}")

        # obtain position
        xcenter, ycenter, width, height = box.xywh[0].tolist()
        plt.scatter(xcenter, ycenter, color=colour, s=40, zorder=5)


def remove_outlier_lines_thermo(lines, width, x_tolerance_mult = 0.25, angle_tolerance = 20):
    """
    Filters the outlier lines that do not match thermometer oriented upright readings.
        - removes non centered lines

    Inputs:
        - lines -- input lines
        - width -- width of the thermometer (should be image width)
        - x_tolerance -- tolerance of x relative positions of the width centred (x_tol * width)
        - angle_tolerance -- tolerance of the angle vertical of 90 degrees (in degrees)
    
    Output:
        - valid_lines -- the valid lines passed these tests
    """
    valid_lines = []
     # get rid of lines that are not centred to the screen (remove outliers)
    center_x = width // 2
    x_tolerance = width * x_tolerance_mult

    for line in lines:
        
        x1, y1, x2, y2 = line[0]
        # Check if both endpoints fall within the x-range of the central fluid line
        relativeX1 = x1 - center_x
        relativeX2 = x2 - center_x
        centeredRule = abs(relativeX1) <= x_tolerance and abs(relativeX2) <= x_tolerance

        # check if line is vertically angled
        dx = x2 - x1
        dy = y2 - y1
        angle =  abs(np.degrees(np.arctan2(dy, dx)))
        angledRule = 90 - angle_tolerance <= angle <= 90 + angle_tolerance

        if centeredRule and angledRule:
            valid_lines.append(line)
        else:
            print(f" [Task 2] Line Removed Not Valid for centred for x end point relative pos, {relativeX1} and {relativeX2}): {centeredRule}, angled vertical: {angledRule}, for angle: {angle}")

    return valid_lines

def get_highest_fluid_endpoint(image, morphed_kernel_size: int, allowDebug: bool):
    """
    Obtains the highest fluid point on the thermometer image.

    Input -> [HSV Threshold for pink fluid] -> [Dilate Fluid] -> [Canny Edge Detection] -> [Hough Lines] -> [Obtain maximum y on the hough lines] -> Output

    Input:
        - image -- the thermometer image
        - morphed_kernel_size -- the size of the kernel
    
    Output:
        - x_point -- the x coordinate of the maximum y detected point
        - y_maximum_point -- the y coorindate of the maximum detected point y
    """
    h, w = image.shape[:2]

    result = hsv_threshold(image, (5, 75, 55), (170, 255, 255)) 
    if allowDebug:
        plt.imshow(result)
        plt.show()
    
    # clean up noise using morphology closed to remove holes 
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (morphed_kernel_size, morphed_kernel_size))
    # dilate so detect lines better
    morphed = cv2.dilate(result, element)

    if allowDebug:
        plt.imshow(morphed)
        plt.show()

    
    gray = cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_LINES_THRESHOLD_FLUID, minLineLength=40, maxLineGap=5)

    # find the top of the resulting region is fluid end point by finding the key point
    if lines is not None:
        y_maximum_point = float("inf") # find the minimum deteted y point (end fluid point 0 is highest)
        x_point = float("inf")
        imageCopy = morphed.copy()

        valid_lines = remove_outlier_lines_thermo(lines, w)
        for line in valid_lines:
            x1, y1, x2, y2 = line[0]
            print(f"Line Segment Endpoints: Start({x1}, {y1}) -> End({x2}, {y2})")
            
            # draw the lines and endpoints on the original image
            cv2.line(imageCopy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # line

            # update maximum y point
            if y2 < y_maximum_point:
                y_maximum_point = y2
                x_point = x2

            if y1 < y_maximum_point:
                y_maximum_point = y1
                x_point = x1

        if allowDebug:
            plt.imshow(imageCopy)
            plt.show()

        return x_point, y_maximum_point
    else:
        print(f"    [Task 2] CANNOT DETECT HOUGH LINES, cannot find fluid endpoint.")
        return 0
    
def run_task2(image_path, config):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # find all the .png images in the input directory
    png_files = sorted(glob.glob(os.path.join(image_path, "*.png")))

    if not png_files:
        print(" [Task 2] No input images found.")
        return

    print(f"    [Task 2] Found {len(png_files)} input images(s).")

    process_task2(png_files, ALLOW_BPM_DEBUG, ALLOW_THERMO_DEBUG)

def process_task2(png_files, allowDebugLCD, allowDebugTHERMO):
    """
    Decides which process to use for lcd and thermometers.

    Input:
        - png_files -- input files to pass
    
    """
    for file_path in png_files:
        fname = os.path.basename(file_path)
        base_name = os.path.splitext(fname)[0] # lcd2 or thermo1

        img = cv2.imread(file_path)
        if img is None or img.size == 0:
            print(f"    [Task 2] Could not read {fname}. Skipping")
            continue

        if fname.startswith("lcd"):
            segment_lcd(img, base_name, allowDebugLCD)
        elif fname.startswith("thermo"):
            segment_thermo(img, base_name, allowDebugTHERMO)
        else:
            print(f"    [Task 2] unknown file type: {fname}. Skipping")