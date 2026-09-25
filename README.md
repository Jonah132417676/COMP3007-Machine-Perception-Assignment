# Synopsis




# Contents
assignment.py - A wrapper script that runs on cmd line inputs to run each task.

task1.py - Detect if in the image has bpm, thermometer or nothing, then crop it and orientate it for its output. 

task2.py - Given a cropped image, you must segment into individual digits (BPM) or segment area on thermometer with fluid mark.

task3.py - Produce predictions on the digit readings or the thermometer readings.

task4.py - Runs the entire pipeline from start to finish.

ml_utils.py - utils script for this project.

config.txt - configs file that contains information about the project and training parameters.

output (directory) - will contain the file outputs of all tasks.

data (directory) - contains necessary models and images required to run the entire pipeline.

# Dependencies
numpy
matplotlib
os
glob
cv2
ultralytics
sys

# Version History
24/09/26 - Initial submission
