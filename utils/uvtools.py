import os
import subprocess
import datetime
import shutil

def extract_layers_with_uvtools(uvtools_exe_path: str, input_slice_file: str, temp_base_folder: str) -> str:
    """
    Executes UVToolsCmd.exe to extract layers from a slice file into a temporary folder.

    Args:
        uvtools_exe_path: The full path to UVToolsCmd.exe.
        input_slice_file: The full path to the .goo or other slice file.
        temp_base_folder: The base directory where temporary files will be stored.

    Returns:
        The path to the folder containing the extracted PNG images.

    Raises:
        FileNotFoundError: If UVToolsCmd.exe or the input file is not found.
        RuntimeError: If the UVTools process returns an error.
        IOError: If the temporary directory cannot be created.
    """
    if not os.path.exists(uvtools_exe_path):
        raise FileNotFoundError(f"UVTools executable not found at: {uvtools_exe_path}")
    if not os.path.exists(input_slice_file):
        raise FileNotFoundError(f"Input slice file not found at: {input_slice_file}")

    if not temp_base_folder:
        raise ValueError("A temporary base folder must be provided.")

    if not os.path.isdir(temp_base_folder):
        try:
            os.makedirs(temp_base_folder, exist_ok=True)
        except OSError as e:
            raise IOError(f"Could not create temporary directory: {temp_base_folder}. Error: {e}")

    # Create a unique subfolder for this session's extraction
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    session_temp_folder = os.path.join(temp_base_folder, f"extraction_{run_timestamp}")
    os.makedirs(session_temp_folder, exist_ok=True)

    command = [
        uvtools_exe_path,
        "extract",
        input_slice_file,
        session_temp_folder,
        "--content",
        "Layers"
    ]

    print(f"Running UVTools command: {' '.join(command)}")

    try:
        # Hide the console window on Windows
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,  # We check the returncode manually
            creationflags=creation_flags,
            encoding='utf-8',
            errors='ignore' # Ignore potential encoding errors from UVTools output
        )

        # UVTools often returns 1 on success with warnings, so we check for codes > 1 as errors
        if process.returncode > 1:
            error_message = f"UVTools exited with error code {process.returncode}:\n" \
                            f"STDOUT: {process.stdout}\n" \
                            f"STDERR: {process.stderr}"
            # Clean up the created temp folder on error
            shutil.rmtree(session_temp_folder)
            raise RuntimeError(error_message)

        print(f"UVTools extraction completed. Output in: {session_temp_folder}")
        return session_temp_folder

    except FileNotFoundError:
        shutil.rmtree(session_temp_folder)
        raise RuntimeError(f"Failed to run UVTools. Is the path correct? Command: {' '.join(command)}")
    except Exception as e:
        # Clean up the created temp folder on any other exception
        shutil.rmtree(session_temp_folder)
        raise RuntimeError(f"An unexpected error occurred while running UVTools: {e}")
