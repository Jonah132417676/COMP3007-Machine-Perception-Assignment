import sys

from ultralytics.utils.torch_utils import strip_optimizer


def print_usage():
    """
    Prints the usage notes for this script.

    """

    print("USAGE: clean_model.py <model_path>")
    print("""
    Example:
        python3 clean_model.py data/task1/bpm_thermometer_detector.pt"
""")
    
if __name__ == "__main__":
    # perform each task
    if len(sys.argv) < 2:
        print("Error: Incorrect use of cmd line")
        print_usage()
        sys.exit(1)


    model_path = sys.argv[1]
        
    # This will remove the extra memory and shrink the file
    strip_optimizer(model_path)
