# src/utils/uvtools.py

import os
import subprocess
import re
import datetime

def run_uvtools_extraction(uvtools_path: str, input_file: str, temp_folder: str, run_timestamp: str) -> str:
    """
    Executes UVToolsCmd.exe to extract layers into a timestamped temp folder.
    """
    session_temp_folder = os.path.join(temp_folder, f"extraction_{run_timestamp}")
    input_folder = os.path.join(session_temp_folder, "Input")

    os.makedirs(input_folder, exist_ok=True)

    command = [
        uvtools_path, "extract", input_file,
        input_folder, "--content", "Layers"
    ]

    try:
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.run(command, capture_output=True, text=True, check=True, creationflags=creation_flags)
        print(f"UVToolsCmd.exe (extract) finished with exit code: {process.returncode}")
        return input_folder
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"UVTools exited with an error (code {e.returncode}):\n\n{e.stderr}")
    except Exception as e:
        raise RuntimeError(f"UVTools extraction failed: {e}")
