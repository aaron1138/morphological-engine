import sys
import os

# Add the 'src' directory to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from gui.main_window import launch_gui

if __name__ == '__main__':
    """
    Main entry point for the Voxel Slice Pre-Processor application.
    """
    print("Launching application...")
    launch_gui()
