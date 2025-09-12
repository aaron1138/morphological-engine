# config.py

import os
import json
from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional, Any
from enum import Enum

DEFAULT_WORKER_COUNT = max(1, os.cpu_count() - 1 if os.cpu_count() else 0)

class OperationType(Enum):
    GPU_SHADER = "GPU Shader"
    ENHANCED_EDT = "Enhanced EDT"
    GAUSSIAN_BLUR = "Gaussian Blur"
    MULTIPLY = "Multiply"
    SCREEN = "Screen"
    OVERLAY = "Overlay"
    APPLY_LUT = "Apply LUT"

class OperationPlane(Enum):
    XY = "XY"
    XZ = "XZ"
    YZ = "YZ"

@dataclass
class LutParameters:
    """Parameters for generating or loading a Look-Up Table (LUT)."""
    lut_source: str = "generated" # or "file"
    lut_generation_type: str = "linear" # linear, gamma, s_curve, etc.
    input_min: int = 0
    input_max: int = 255
    output_min: int = 0
    output_max: int = 255
    gamma_value: float = 1.0
    s_curve_contrast: float = 0.5
    fixed_lut_path: str = ""
    spline_points: List[List[int]] = field(default_factory=lambda: [[0, 0], [255, 255]])

@dataclass
class PipelineOperation:
    """Represents a single operation in the processing pipeline."""
    type: str = OperationType.ENHANCED_EDT.value
    plane: str = OperationPlane.XY.value

    # --- Parameters for Enhanced EDT ---
    look_forward: int = 3
    look_backward: int = 3
    fade_distance_limit: float = 10.0

    # --- Parameters for GPU Shader ---
    shader_file: str = ""

    # --- Parameters for Gaussian Blur ---
    gaussian_ksize_x: int = 5

    # --- Parameters for Blending ---
    blend_mode: str = ""

    # --- Parameters for LUT ---
    lut_params: LutParameters = field(default_factory=LutParameters)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        # A simple way to load, will need to be more robust if new
        # fields are added that are not in older configs.
        obj = cls()
        for f in fields(cls):
            if f.name in data:
                setattr(obj, f.name, data[f.name])
        return obj

@dataclass
class Config:
    """Main application configuration."""
    # --- I/O Settings ---
    input_mode: str = "folder"
    input_folder: str = ""
    output_folder: str = ""
    uvtools_path: str = "C:\\Program Files\\UVTools\\UVToolsCmd.exe"
    uvtools_temp_folder: str = ""
    uvtools_input_file: str = ""
    uvtools_cleanup: bool = True

    # --- General Settings ---
    worker_count: int = DEFAULT_WORKER_COUNT
    use_numba: bool = True

    # --- Pipeline ---
    pipeline: List[PipelineOperation] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['pipeline'] = [op.to_dict() for op in self.pipeline]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        config_instance = cls()
        config_fields = {f.name for f in fields(cls)}

        for key, value in data.items():
            if key in config_fields:
                if key == 'pipeline':
                    config_instance.pipeline = [PipelineOperation.from_dict(op_data) for op_data in value]
                else:
                    setattr(config_instance, key, value)
        return config_instance

    def save(self, filepath: str):
        """Saves the current configuration to a JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=4)
        except Exception as e:
            print(f"Error saving configuration to {filepath}: {e}")

    @classmethod
    def load(cls, filepath: str) -> "Config":
        """Loads configuration from a JSON file."""
        if not os.path.exists(filepath):
            return cls() # Return default config if file doesn't exist
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error decoding configuration file {filepath}: {e}. Returning default config.")
            return cls()
        except Exception as e:
            print(f"An unexpected error occurred while loading config {filepath}: {e}. Returning default config.")
            return cls()


# --- Global application config instance ---
CONFIG_FILE_PATH = "app_config.json"
app_config = Config.load(CONFIG_FILE_PATH)
