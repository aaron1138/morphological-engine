# -*- coding: utf-8 -*-
"""
Module: rawgl_pipeline.py
Author: Jules
Description: A pipeline controller for executing the external RawGL command-line tool.
"""

import subprocess
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

    def run(self, input_path: str, output_path: str):
        """
        Constructs and executes the RawGL command for a single image.

        Args:
            input_path (str): The path to the source image file.
            output_path (str): The path to save the processed image file.

        Raises:
            subprocess.CalledProcessError: If RawGL returns a non-zero exit code.
        """
        shader_path = self.config["shader_path"]

        # Build the command as a list of arguments
        command = [
            str(self.executable_path),
            # Specify the compute shader
            '-C', str(shader_path),
            # Specify the input texture (uniform name 'Texture0' is a common convention)
            '-i', 'Texture0', str(input_path),
            # Specify the output file
            '-o', 'OutColor', str(output_path),
            # Enforce single-channel, 8-bit output for greyscale PNG
            '-n', '1', # Number of channels
            '-b', '8', # Bits per channel
        ]

        print(f"Executing RawGL command: {' '.join(command)}")

        try:
            # Execute the command
            result = subprocess.run(
                command,
                check=True,          # Raise an exception for non-zero exit codes
                capture_output=True, # Capture stdout and stderr
                text=True            # Decode stdout/stderr as text
            )
            print(f"RawGL processing successful for {Path(input_path).name}.")
            if result.stdout:
                print(f"RawGL stdout:\n{result.stdout}")
            if result.stderr:
                # RawGL often prints info to stderr, so we just print it.
                print(f"RawGL stderr:\n{result.stderr}")

        except FileNotFoundError:
            print(f"ERROR: Could not find the RawGL executable at '{self.executable_path}'.")
            raise
        except subprocess.CalledProcessError as e:
            print(f"ERROR: RawGL process failed for {Path(input_path).name} with exit code {e.returncode}.")
            print(f"  Stderr: {e.stderr}")
            print(f"  Stdout: {e.stdout}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while running RawGL: {e}")
            raise
