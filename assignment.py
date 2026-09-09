"""

assignment.py

A wrapper script that handles running tests on task1, task2, task3 and task4 modules.

Author: Zhong Cheng Lau 

Last Modified: 2026-09-09

"""
import sys
import os

from task1 import run_task1
from task2 import run_task2
from task3 import run_task3
from task4 import run_task4

# read the config file 
def read_config(config_path):
    """
    Reads the config file and outputs dictionary with keywords.
    """
    config = {}
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return config

    with open(config_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                key, value = line.split(':')
                config[key.strip()] = value.strip()
    return config


def print_usage():
    """
    Prints the usage notes for this script.

    """

    print("USAGE: assignment.py <task> <image_path> [config_path]")
    print("""
    Example:
        python3 assignment.py task1 data/validation/task3/lcd2/d1.png"
        Optional config file path (default: config.txt) in same directory
        python3 assignment.py task1 data/validation/task3/lcd2/d1.png ./config.txt
""")

if __name__ == "__main__":
    # perform each task
    if len(sys.argv) < 3:
        print("Error: Incorrect use of cmd line")
        print_usage()
        sys.exit(1)


    task = sys.argv[1]
    image_path = sys.argv[2]
    config_path = sys.argv[3] if len(sys.argv) > 3 else 'config.txt' # default to config.txt if not provided

    config = read_config(config_path)

    if not config:
        sys.exit(1)

    try:   
        match task:
            case "task1":
                run_task1(image_path, config) 
            case "task2":
                run_task2(image_path, config)
            case "task3":
                run_task3(image_path, config)
            case "task4":
                run_task4(image_path, config)
            case _: # default case
                print(f"Unknown task: {task}. Specify task1, task2, task3 or task4")
                print_usage()
                sys.exit(1)

    except Exception as e:
        print(f"An error occurred while executing {task}: {e}")
        sys.exit(1)
    