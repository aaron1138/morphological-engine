# -*- coding: utf-8 -*-
"""
Module: rawgl_controller.py
Author: Jules
Description: Provides a Python interface to the RawGL command-line executable.
             This controller builds the necessary command-line arguments from a
             configuration object and executes RawGL as a subprocess.
"""

import subprocess
import shlex
from typing import List, Tuple
from pathlib import Path

# Use the new dataclass for configuration
from .rawgl_config import RawGLConfig

class RawGLController:
    """
    Manages the execution of the RawGL external executable.
    """
    def __init__(self, executable_path: str):
        """
        Initializes the controller with the path to the rawgl.exe.

        Args:
            executable_path (str): The full path to the rawgl.exe file.

        Raises:
            FileNotFoundError: If the executable is not found at the given path.
        """
        self.executable_path = Path(executable_path)
        if not self.executable_path.is_file():
            raise FileNotFoundError(f"RawGL executable not found at: {self.executable_path}")

    def build_command(self, config: RawGLConfig) -> List[str]:
        """
        Builds a list of command-line arguments from a RawGLConfig object.

        Args:
            config (RawGLConfig): A dataclass object describing the RawGL job.

        Returns:
            List[str]: A list of strings representing the command-line arguments.
        """
        command = []

        # Compute pass
        command.extend(['-C', str(config.shader_path)])

        # Input texture (hardcode uniform name "Texture0" for simplicity)
        command.extend(['-i', 'Texture0', str(config.input_path)])

        # Output texture (hardcode uniform name "OutColor" for simplicity)
        command.extend(['-o', 'OutColor', str(config.output_path)])

        # Pass size
        if config.output_size:
            command.extend(['-S', str(config.output_size[0]), str(config.output_size[1])])
        else:
            # Special syntax to use the size of the input texture
            command.extend(['-S', 'Texture0::0', 'Texture0::1'])

        # Workgroup size
        command.extend(['-W', str(config.workgroup_size[0]), str(config.workgroup_size[1])])

        # Add fixed arguments for single-channel 8-bit grayscale PNGs
        command.extend(['--out_format', 'r8'])
        command.extend(['--out_channels', '1'])
        command.extend(['--out_bits', '8'])

        return command

    def run(self, config: RawGLConfig) -> Tuple[str, str]:
        """
        Builds the command and runs the RawGL executable as a subprocess.

        Args:
            config (RawGLConfig): The configuration for the RawGL job.

        Returns:
            A tuple containing the stdout and stderr from the process.

        Raises:
            RuntimeError: If the RawGL process returns a non-zero exit code.
        """
        command_args = self.build_command(config)
        full_command = [str(self.executable_path)] + command_args

        print(f"Executing RawGL command: {' '.join(shlex.quote(arg) for arg in full_command)}")

        process = subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            error_message = (
                f"RawGL execution failed with exit code {process.returncode}.\n\n"
                f"Command: {' '.join(shlex.quote(arg) for arg in full_command)}\n\n"
                f"Stderr:\n{stderr}"
            )
            raise RuntimeError(error_message)

        return stdout, stderr
