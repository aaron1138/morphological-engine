# -*- coding: utf-8 -*-
"""
Module: rawgl_config.py
Author: Jules
Description: Defines the data structure for configuring a RawGL processing job.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path

@dataclass
class RawGLConfig:
    """
    Represents the configuration for a single RawGL processing job,
    focusing on a single compute shader pass for image processing.
    """
    shader_path: Path
    input_path: Path
    output_path: Path
    workgroup_size: Tuple[int, int] = field(default_factory=lambda: (8, 8))
    output_size: Optional[Tuple[int, int]] = None  # If None, will default to input size.

    def __post_init__(self):
        """Validate paths after initialization."""
        if not self.shader_path.is_file():
            raise FileNotFoundError(f"Shader file not found: {self.shader_path}")
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        # output_path is a destination, so we check its parent directory.
        if not self.output_path.parent.is_dir():
            raise NotADirectoryError(f"Output directory does not exist: {self.output_path.parent}")
