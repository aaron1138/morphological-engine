import os
import subprocess

def extract_layers(uvtools_path: str, input_file: str, temp_folder: str) -> str:
    """
    Executes UVToolsCmd.exe to extract layers into a timestamped temp folder.
    Returns the path to the folder containing the extracted images.
    """
    input_folder = os.path.join(temp_folder, "Input")
    os.makedirs(input_folder, exist_ok=True)

    command = [uvtools_path, "extract", input_file, input_folder, "--content", "Layers"]

    try:
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        # Remove check=True to handle non-zero exit codes manually
        process = subprocess.run(command, capture_output=True, text=True, creationflags=creation_flags)
        # A return code of 1 is sometimes expected and not a fatal error.
        if process.returncode not in [0, 1]:
            error_message = f"UVTools exited with an unexpected error (code {process.returncode}):\n\n{process.stderr}"
            raise RuntimeError(error_message)
        return input_folder
    except subprocess.CalledProcessError as e:
        # This block is now less likely to be hit, but good practice to keep
        error_message = f"UVTools extraction failed. Command: '{' '.join(e.cmd)}'\n"
        error_message += f"Return Code: {e.returncode}\n"
        error_message += f"Stderr: {e.stderr}\n"
        error_message += f"Stdout: {e.stdout}\n"
        raise RuntimeError(error_message)
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during UVTools extraction: {e}")
