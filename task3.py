"""

task3.py

Produce predictions on the digit readings or the thermometer readings.


Author: Zhong Cheng Lau 

Last Modified: 2026-20-09

"""

import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

from ml_utils import load_YOLO, hsv_threshold
from task2 import get_highest_fluid_endpoint

ALLOW_BPM_DEBUG = False
ALLOW_THERMO_DEBUG = True

OUTPUT_DIR = os.path.join("output","task3")
LCD_MODEL_PATH = os.path.join("data/task3/digit_LCD_classifier_model/","lcd_digit_detector.pt")

MORPHED_KERNEL_SIZE = 1
LEFT_THERMO_TICK_THRESHOLD = 0.3
RIGHT_THERMO_TICK_THRESHOLD = 0.7
Y_TICKS_GROUPING_THRESHOLD = 8

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

# pad size trial and error for the model
# 450
def recognise_lcd_digits(sub_dir_path, lcd_name, pad_size = 450, padding_color = [103, 140, 160]):
    """
    Recognise the digit from the image. Additional processing of digit images include padding and setting padding color, which helps the model to identify the digits better.
    
    improve task
    3 lcd reading, so it segments the lcd digit, th
    en pads to the background coluor (so its strong
    er)
    Pipeline: Input -> [Yolov8 Digit Detection] -> Output
    Input:
        - sub_dir_path -- path to the subdirectory
        - lcd_name -- name of lcd image
        - pad_size -- padding applied for the model to detect digits better
        - padding_color -- affects model learning and makes the background a more homogenous color
    """
    out_dir = os.path.join(OUTPUT_DIR, lcd_name)

    digit_files = sorted(glob.glob(os.path.join(sub_dir_path, "d*.png")))
    yolo = load_YOLO(LCD_MODEL_PATH)

    if not digit_files:
        print(f"    [Task 3] {lcd_name}: no digit images found.")

    # for each digit image input
    for dfile in digit_files:   
        img = cv2.imread(dfile)
        # add padding as model has trained on smaller lcd digits, not full
        padded_img = cv2.copyMakeBorder(
            img, 
            top=pad_size, bottom=pad_size, left=pad_size, right=pad_size, 
            borderType=cv2.BORDER_CONSTANT, 
            value=padding_color
            
        )
        
        dname = os.path.splitext(os.path.basename(dfile))[0] # d1, d2 ...

        # obtain the singular result
        result = yolo.predict(padded_img, retina_masks = True)[0]
        class_names = result.names

        if ALLOW_BPM_DEBUG:
            plt.imshow(result.plot())
            plt.show()

        if result is None or len(result.boxes) == 0:
            # negative produce no output
            final_image = None
            print(f"    [Task 3] {dname}.jpg -> (no output)")
            continue

        # set default digit
        digit = -1
        # identify digit got the boxes
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            # obtain label name
            label_name = class_names[class_id]
            # for each image if negative 
            print(f"LABEL: {label_name}, CONF: {box.conf[0].item()}")

            digit = label_name

            save_output(os.path.join(out_dir, f"{dname}.txt"), str(digit), output_type="txt")

    print(f"    [Task 3] {lcd_name}: recognised {len(digit_files)} digit(s) -> {out_dir}")

def calculate_temperature(sub_dir_path, thermo_name):
    """
    Calculates the temperature from the reading.

    Pipeline: Input -> [HSV Color Space] -> [Threshold Pink/Fluid Endpoint and obtain y position] -> [Canny Edge detector on normal image] -> [Hough lines detection] -> [Find average y distance between ticks] -> [Find the digit value from left most, bottom most tick number detector] -> [Find y distance] -> [Find how many ticks yDistance * ticks/ydist] -> [Then calculate temperature reading] -> output

    Input:
        - sub_dir_path -- path of the sub directory of the input image.
        - thermo_name -- name of the thermoemter file
    
    """
    out_dir = os.path.join(OUTPUT_DIR, thermo_name)

    # check that t.png exists
    t_path = os.path.join(sub_dir_path, "t.png")
    if not os.path.isfile(t_path):
        print(f"    [Task 3] {thermo_name} : t.png not found. Skipping.")
        return

    # load temperature image
    image = cv2.imread(t_path)
    h, w = image.shape[:2]
    # find fluid end point position (y).
    # convert to hsv (hue, sat, val) space img 
    x, y_maximum_point = get_highest_fluid_endpoint(image, MORPHED_KERNEL_SIZE, ALLOW_THERMO_DEBUG)
    
    if ALLOW_THERMO_DEBUG:
        imgCopy = image.copy()
        cv2.circle(imgCopy, (x , y_maximum_point), 3, color=[255, 0, 0])
        plt.imshow(imgCopy)
        plt.show()
  

    # find all x tick   s (line detection)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # isolate black ticks to remove other details lines
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(mask)
    plt.show()

    # remove sides (numbers may detect lines so remove quuarter from either side)
    left_bound = int(w * LEFT_THERMO_TICK_THRESHOLD)
    right_bound = int(w * RIGHT_THERMO_TICK_THRESHOLD)
    mask[:, 0:left_bound] = 0
    mask[:, right_bound:w] = 0    
    plt.imshow(mask)
    plt.show()


    lines = cv2.HoughLinesP(
        image=mask, 
        rho=1, 
        theta=np.pi / 360, 
        threshold=70, # lower threshold as zoomed in more
        )
        
   
    # make sure lines are horizontal relatively (not 1) and on left side readings
    if lines is not None:
        imgCopy = image.copy()
        valid_lines = filter_horizontal_lines(lines, 5)

        for line in valid_lines:
            x1, y1, x2, y2 = line[0]


            cv2.line(imgCopy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.imshow(imgCopy)
        plt.show()

        # obtain the average y distance per tick
        avgYDistPerTick = calculate_average_y_dist_per_tick(valid_lines, image)

        # find the temperature reading on the left side of thermometer, remove set 0 the right side

        # then calculate temperature
        t_base = 10 # temperature reading below
        yDifference = 237 - y_maximum_point
        ticks = yDifference / avgYDistPerTick # conversion

        # round to closest int
        isFluidAboveReading = True
        ticksMult = 1 if isFluidAboveReading else -1 # if fluid is above reading, add, else subtract as its below
        # temperature = base reading + ticksCounted * add or subtract
        temperature = round(t_base + ticks * ticksMult)
    
        save_output(os.path.join(out_dir, "t.txt"), str(temperature), output_type="txt")    
        print(f"    [Task 3] {thermo_name}: esimated temperature = {temperature}C, y_maximum_point{y_maximum_point}")

    else:
        print("     [Task 3] Thermo lines not detected ticks.")
         



   

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

def filter_horizontal_lines(lines, angle_tolerance):
    
    valid_lines = []
  

    for line in lines:
        
        x1, y1, x2, y2 = line[0]
        
        # check if line is vertically angled
        dx = x2 - x1
        dy = y2 - y1
        angle =  abs(np.degrees(np.arctan2(dy, dx)))
        angledRule =  -angle_tolerance <= angle <= angle_tolerance

        if angledRule:
            valid_lines.append(line)
        else:
            print(f" [Task 3] Line Removed Not Valid for angled horizontal: {angledRule}, for angle: {angle}")

    return valid_lines

def calculate_average_y_dist_per_tick(lines, image):

    # to calculate ticks per y, find the average distance between each tick
    # find y positions of each line
    yPositions = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        yAvg = (y1 + y2)/2
        yPositions.append(yAvg)

    # sort by growing order of y's
    yPositions.sort(key=lambda y: y) 

    # since there is cluster position ups in y positions, obtain the average of close double ups
    cleanedYPositions = group_similar_y(yPositions, Y_TICKS_GROUPING_THRESHOLD)

    imgCopy = image.copy()
    h, w = imgCopy.shape[:2]

    for y in cleanedYPositions:
        cv2.circle(imgCopy, center=(int(w//2), int(y)), radius=5, color=[255, 0, 0])
    plt.imshow(imgCopy)
    plt.show()

    avgYDistBetweenTicks = 0
    for i in range(0, len(cleanedYPositions) - 1):
        avgYDistBetweenTicks += abs(cleanedYPositions[i] - cleanedYPositions[i + 1]) # difference between sorted

    avgYDistBetweenTicks /= len(cleanedYPositions)
    print(f"     [Task 3] Thermo calculated average y dist between ticks: {avgYDistBetweenTicks}")
    return avgYDistBetweenTicks
    


def group_similar_y(yPositionsSorted, groupingThreshold):

    result = []
    currentGroup = [yPositionsSorted[0]]
    for i in range(1, len(yPositionsSorted)):
        # check neighbouring list elements of there are similar within threshold
        # if similar add to current Group, else average and add to result
        y1 = yPositionsSorted[i]
        y0 = yPositionsSorted[i - 1]
        if abs(y1 - y0) < groupingThreshold: # add to group
            currentGroup.append(y1)
        else:
            average = np.array(currentGroup).mean()
            result.append(average)

            # clear current group
            currentGroup = []

    # add result final if it not empty
    if len(currentGroup) > 0:
        average = np.array(currentGroup).mean()
        result.append(average)

    return result

