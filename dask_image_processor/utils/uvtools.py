# utils/uvtools.py

import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_layers(uvtools_path: str, input_file: str, output_dir: str) -> str:
    """
    Extracts layers from a slice file using UVToolsCMD.exe.

    Args:
        uvtools_path: The full path to UVToolsCmd.exe.
        input_file: The path to the input slice file (e.g., .ctb, .goo).
        output_dir: The directory where the PNG layers should be saved.

    Returns:
        The path to the directory containing the extracted images.

    Raises:
        FileNotFoundError: If uvtools_path or input_file does not exist.
        subprocess.CalledProcessError: If the UVTools command fails.
        RuntimeError: If the expected output directory is not created.
    """
    if not os.path.exists(uvtools_path):
        raise FileNotFoundError(f"UVTools executable not found at: {uvtools_path}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input slice file not found at: {input_file}")

    os.makedirs(output_dir, exist_ok=True)

    # UVTools automatically creates a subfolder named after the input file
    input_filename_base = os.path.splitext(os.path.basename(input_file))[0]
    expected_output_path = os.path.join(output_dir, input_filename_base)

    logging.info(f"Starting UVTools extraction for {input_file} into {output_dir}")

    command = [
        uvtools_path,
        "unpack",
        "--to-dir", output_dir,
        input_file
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=subprocess.CREATE_NO_WINDOW)

        # Log output in real-time
        for line in iter(process.stdout.readline, ''):
            logging.info(f"UVTools: {line.strip()}")

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

    except subprocess.CalledProcessError as e:
        error_message = f"UVTools extraction failed with exit code {e.returncode}."
        logging.error(error_message)
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during UVTools extraction: {e}")
        raise

    if not os.path.isdir(expected_output_path):
        raise RuntimeError(f"UVTools did not create the expected output directory: {expected_output_path}")

    logging.info(f"UVTools extraction successful. Images are in: {expected_output_path}")
    return expected_output_path


def repack_layers(*args, **kwargs):
    """
    Placeholder for repacking layers.

    A full implementation would require creating a .uvtop file and calling
    the 'uvtools apply' command. This is a complex task that is beyond the
    scope of the initial implementation.
    """
    logging.warning("repack_layers is not yet implemented.")
    # Example of what the command might look like:
    # command = [
    #     uvtools_path,
    #     "apply",
    #     "--operation-stack", uvtop_file,
    #     "--output", output_file,
    #     original_file
    # ]
    pass
