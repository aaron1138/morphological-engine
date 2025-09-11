# -*- coding: utf-8 -*-
"""
Module: rawgl_pipeline.py
Author: Jules (Refactored for Dask)
Description: A pipeline controller for executing the external RawGL command-line tool
             on in-memory NumPy arrays provided by a Dask graph.
"""

import subprocess
import tempfile
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any

class RawGlPipeline:
    """
    Manages and executes a processing pipeline using an external RawGL executable.
    """
    def __init__(self, config: Dict[str, Any], rawgl_executable_path: str):
        """
        Initializes the RawGL pipeline controller.

        Args:
            config (Dict[str, Any]): A dictionary defining the RawGL parameters.
            rawgl_executable_path (str): The full path to the rawgl.exe file.
        """
        self.config = config
        self.executable_path = Path(rawgl_executable_path)
        self._validate_config()

    def _validate_config(self):
        """Validates the structure of the pipeline configuration."""
        if not self.executable_path.is_file():
            raise FileNotFoundError(f"RawGL executable not found at: {self.executable_path}")
        if "shader_path" not in self.config:
            raise ValueError("Configuration must contain a 'shader_path'.")
        shader_path = Path(self.config["shader_path"])
        if not shader_path.is_file():
            raise FileNotFoundError(f"Shader file not found at: {shader_path}")

    def run(self, slice_data: np.ndarray, output_path: str):
        """
        Takes an in-memory NumPy array, saves it to a temporary file,
        and executes the RawGL command on it.

        Args:
            slice_data (np.ndarray): The 2D image data for a single slice.
            output_path (str): The path to save the processed image file.
        """
        shader_path = self.config["shader_path"]

        # Use a temporary file for the input, ensuring it's deleted afterward
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_input_file:
            input_temp_path = temp_input_file.name

            # Save the in-memory numpy array to the temporary PNG file
            cv2.imwrite(input_temp_path, slice_data)

            # Build the command
            command = [
                str(self.executable_path),
                '-C', str(shader_path),
                '-i', 'Texture0', str(input_temp_path),
                '-o', 'OutColor', str(output_path),
                '-n', '1',
                '-b', '8',
            ]

            try:
                # Execute the command
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True
                )
            except FileNotFoundError:
                raise
            except subprocess.CalledProcessError as e:
                # Re-raise with more context
                error_message = (
                    f"RawGL failed for input derived from slice data.\n"
                    f"Exit Code: {e.returncode}\n"
                    f"Stderr: {e.stderr}\n"
                    f"Stdout: {e.stdout}"
                )
                raise RuntimeError(error_message) from e
            except Exception as e:
                raise RuntimeError(f"An unexpected error occurred while running RawGL: {e}") from e

        # print(f"RawGL processing successful for {output_path}") # Too noisy for parallel execution
