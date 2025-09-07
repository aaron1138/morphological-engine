# -*- coding: utf-8 -*-
"""
Module: run_logger.py
Author: Gemini
Description: Logs the details of each processing run to a JSON file,
             creating a simple "run database".
"""

import json
import datetime
from pathlib import Path
from typing import Dict, Any, List

class RunLogger:
    """
    Handles logging of processing runs to a persistent JSON file.
    """
    def __init__(self, log_file_path: str = 'run_log.json'):
        """
        Initializes the logger with a path to the log file.

        Args:
            log_file_path (str): The name of the log file. It will be created
                                 in the application's root directory.
        """
        self.log_file = Path(log_file_path)

    def _read_log(self) -> List[Dict[str, Any]]:
        """Reads the entire log file into a list of dictionaries."""
        if not self.log_file.exists():
            return []
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If the file is corrupted or unreadable, start fresh
            return []

    def _write_log(self, log_data: List[Dict[str, Any]]):
        """Writes the list of dictionaries to the log file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=4)
        except IOError as e:
            print(f"Error: Could not write to run log file: {e}")

    def log_run(self, run_details: Dict[str, Any]):
        """
        Adds a new run's details to the log.

        Args:
            run_details (Dict[str, Any]): A dictionary containing all the
                                          relevant data for the processing run.
                                          A timestamp will be added automatically.
        """
        logs = self._read_log()

        # Add a timestamp and nest the original data for clarity
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'run_details': run_details
        }

        logs.append(log_entry)
        self._write_log(logs)
        print(f"Successfully logged run to {self.log_file}")

# --- Example Usage ---
if __name__ == '__main__':
    print("--- RunLogger Test ---")

    # Create a logger instance (will create run_log.json in the current dir)
    logger = RunLogger(log_file_path='temp_run_log.json')

    # Define some dummy run data
    run1_data = {
        "input_directory": "C:/slices/project_a",
        "output_directory": "C:/slices/project_a/output",
        "settings": {
            "window_size": 5,
            "steps": [
                {"name": "morph_open", "params": {"kernel_size": 3}},
                {"name": "blur", "params": {"sigma": 1.5}}
            ]
        },
        "shader_used": "shaders/morphological.comp",
        "duration_seconds": 125.5,
        "status": "Completed"
    }

    run2_data = {
        "input_directory": "D:/other_slices/project_b",
        "output_directory": "D:/other_slices/project_b/output",
        "settings": {
            "window_size": 3,
            "steps": [
                {"name": "blur", "params": {"sigma": 2.0}}
            ]
        },
        "shader_used": "shaders/blur.comp",
        "duration_seconds": 55.1,
        "status": "Failed",
        "error_message": "Disk full."
    }

    # Log the runs
    print("\nLogging first run...")
    logger.log_run(run1_data)

    print("\nLogging second run...")
    logger.log_run(run2_data)

    # Verify the log file content
    print("\nVerifying log file content...")
    log_content = logger._read_log()
    if len(log_content) == 2 and log_content[1]['status'] == 'Failed':
        print("  -> Success! Log file contains 2 entries and content seems correct.")
    else:
        print("  -> Failure! Log file content is not as expected.")

    # Clean up the test file
    if Path('temp_run_log.json').exists():
        Path('temp_run_log.json').unlink()
        print("\nCleaned up temp_run_log.json")
