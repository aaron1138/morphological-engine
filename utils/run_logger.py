# -*- coding: utf-8 -*-
"""
Module: run_logger.py
Author: Jules
Description: Provides a simple "run database" by logging processing job
             details to a JSON Lines file.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

def log_run(
    run_data: Dict[str, Any],
    log_filename: str = "runs.jsonl"
):
    """
    Appends a record of a processing run to a JSON Lines log file.

    Each call to this function appends a single line (a self-contained
    JSON object) to the log file.

    Args:
        run_data (Dict[str, Any]): A dictionary containing the data for the
                                   processing run. Must be JSON-serializable.
        log_filename (str): The name of the log file. It will be created in
                            the project's root directory.
    """
    log_file = Path(log_filename)

    # Ensure the run data includes a timestamp
    if "timestamp" not in run_data:
        run_data["timestamp"] = datetime.now(timezone.utc).isoformat()

    try:
        # Open the file in append mode, creating it if it doesn't exist
        with open(log_file, 'a') as f:
            # Serialize the dictionary to a JSON string
            log_entry = json.dumps(run_data)
            # Write the JSON string as a new line
            f.write(log_entry + '\n')
        print(f"Successfully logged run to '{log_file}'.")
    except (IOError, TypeError) as e:
        # TypeError could happen if run_data is not JSON-serializable
        print(f"Error logging run to '{log_file}': {e}")

# --- Example Usage ---
if __name__ == '__main__':
    print("--- RunLogger Test ---")

    # Define some mock data for a processing run
    mock_processing_config = {
        "steps": [
            {"operation": "gradient_sobel", "threshold": 128},
            {"operation": "dilate", "kernel_size": 3}
        ]
    }

    mock_run_info = {
        "input_directory": "C:/projects/test_data/slices_01",
        "output_directory": "C:/projects/test_data/output_01",
        "gpu_used": "NVIDIA GeForce RTX 4090 (Device ID 0)",
        "processing_time_seconds": 125.7,
        "processing_config": mock_processing_config,
        "user_notes": "Test run with new Sobel threshold."
    }

    # Log the run
    log_run(mock_run_info)

    # Log another run
    mock_run_info_2 = mock_run_info.copy()
    mock_run_info_2["input_directory"] = "C:/projects/test_data/slices_02"
    mock_run_info_2["user_notes"] = "Second run with same settings."
    log_run(mock_run_info_2)

    print("\nCheck the 'runs.jsonl' file for the logged data.")
