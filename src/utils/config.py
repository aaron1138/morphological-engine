# src/utils/config.py

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Any
import json
import os

DEFAULT_NUM_WORKERS = max(1, os.cpu_count() - 1)

@dataclass
class Config:
    """
    Main application configuration.
    """
    # --- I/O Settings ---
    input_mode: str = "folder"  # "folder" or "uvtools"
    input_folder: str = ""
    output_folder: str = ""
    start_index: Optional[int] = 0
    stop_index: Optional[int] = None

    # --- UVTools Mode Settings ---
    uvtools_path: str = "C:\\Program Files\\UVTools\\UVToolsCmd.exe"
    uvtools_temp_folder: str = ""
    uvtools_input_file: str = ""
    uvtools_output_location: str = "working_folder"
    uvtools_delete_temp_on_completion: bool = True
    output_file_prefix: str = "Dask_Processed_"

    # --- Dask Settings ---
    dask_chunk_size_x: int = 64
    dask_chunk_size_y: int = 64
    dask_chunk_size_z: int = 64
    dask_thread_count: int = DEFAULT_NUM_WORKERS

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        config_instance = cls()
        for key, value in data.items():
            if hasattr(config_instance, key):
                setattr(config_instance, key, value)
        return config_instance

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> "Config":
        if not os.path.exists(filepath):
            default_config = cls()
            try:
                default_config.save(filepath)
            except Exception as e:
                print(f"Error saving default config: {e}")
            return default_config
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            print(f"Error loading config '{filepath}': {e}. Using default.")
            return cls()

_CONFIG_FILE = "app_config.json"
app_config = Config.load(_CONFIG_FILE)
