# -*- coding: utf-8 -*-
"""
Module: rawgl_controller.py
Author: Gemini
Description: Manages the execution of RawGL for shader-based image processing.
"""

import subprocess
import cv2
import os
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

class RawGLController:
    """
    A controller for building command-line arguments and running the RawGL
    executable in a multi-threaded fashion.
    """
    def __init__(self, rawgl_path: str, max_workers: int = None):
        """
        Initializes the RawGLController.

        Args:
            rawgl_path (str): The full path to the rawgl.exe executable.
            max_workers (int, optional): The maximum number of concurrent
                                         processes. Defaults to the number of CPU cores.
        """
        if not Path(rawgl_path).exists():
            raise FileNotFoundError(f"RawGL executable not found at: {rawgl_path}")
        self.rawgl_path = rawgl_path
        self.max_workers = max_workers if max_workers else os.cpu_count()

    def _get_image_dimensions(self, image_path: Path) -> Optional[Tuple[int, int]]:
        """Gets the width and height of an image file."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            height, width, _ = img.shape
            return width, height
        except Exception as e:
            print(f"Error reading image dimensions for {image_path}: {e}")
            return None

    def process_image(self, input_path: Path, output_path: Path, shader_path: Path, channels: int, bits: int) -> Tuple[bool, str]:
        """
        Constructs and executes a RawGL command for a single image.

        Args:
            input_path (Path): Path to the input image.
            output_path (Path): Path to save the processed image.
            shader_path (Path): Path to the compute shader file.
            channels (int): Number of output channels (e.g., 1 for greyscale).
            bits (int): Bit depth of the output image (e.g., 8 or 16).

        Returns:
            A tuple (success, message).
        """
        dims = self._get_image_dimensions(input_path)
        if not dims:
            return False, f"Could not get dimensions for {input_path.name}"

        cmd = [
            self.rawgl_path,
            '--pass_comp', str(shader_path),
            '--pass_size', str(dims[0]), str(dims[1]),
            '--in', 'InputTex', str(input_path),
            '--out', 'OutputColor', str(output_path),
            '--out_channels', str(channels),
            '--out_bits', str(bits)
        ]

        try:
            # Using DEVNULL to hide RawGL's console output for a cleaner UI experience
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if result.returncode == 0:
                return True, f"Successfully processed {input_path.name}"
            else:
                return False, f"Error processing {input_path.name}:\n{result.stderr}"
        except subprocess.CalledProcessError as e:
            return False, f"RawGL execution failed for {input_path.name}:\n{e.stderr}"
        except FileNotFoundError:
            return False, "rawgl.exe not found. Please check the path in settings."
        except Exception as e:
            return False, f"An unexpected error occurred for {input_path.name}: {e}"

    def process_batch(self, image_paths: List[Path], output_dir: Path, shader_path: Path, channels: int, bits: int, progress_callback=None):
        """
        Processes a batch of images using a thread pool.

        Args:
            image_paths (List[Path]): List of input image paths.
            output_dir (Path): Directory to save processed images.
            shader_path (Path): Path to the compute shader.
            channels (int): Number of output channels.
            bits (int): Bit depth of the output.
            progress_callback (function, optional): A function to call with progress updates.
        """
        output_dir.mkdir(exist_ok=True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_image, img_path, output_dir / img_path.name, shader_path, channels, bits): img_path
                for img_path in image_paths
            }

            for i, future in enumerate(as_completed(futures)):
                img_path = futures[future]
                try:
                    success, message = future.result()
                    if not success:
                        print(message) # Log errors to console
                except Exception as e:
                    print(f"An error occurred while processing {img_path.name}: {e}")

                if progress_callback:
                    progress_callback(i + 1, len(image_paths))

if __name__ == '__main__':
    print("--- RawGLController Test ---")
    # This test requires a dummy rawgl executable and some test images/shaders.
    # For now, we are just ensuring the class structure is sound.

    # Create a dummy executable for testing purposes
    dummy_exe_path = Path("dummy_rawgl.exe")
    with open(dummy_exe_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'Dummy RawGL executed with args: $@'\n")
        f.write("exit 0\n")
    os.chmod(dummy_exe_path, 0o755)

    try:
        controller = RawGLController(rawgl_path=str(dummy_exe_path))
        print(f"Controller initialized with path: {controller.rawgl_path}")
        print(f"Using max workers: {controller.max_workers}")

        # Test command generation (without running subprocess)
        # This part is implicitly tested in the process_image method,
        # but a dedicated test could be added in the pytest suite.
        print("\nThis module is best tested through the application's processing thread.")

    except FileNotFoundError as e:
        print(e)
    finally:
        # Clean up dummy file
        if dummy_exe_path.exists():
            dummy_exe_path.unlink()
            print(f"\nCleaned up {dummy_exe_path}")
    print("--- Test Complete ---")
