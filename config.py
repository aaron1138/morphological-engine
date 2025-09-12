import json
import os
from dataclasses import dataclass, field, asdict

DEFAULT_NUM_WORKERS = max(1, os.cpu_count() // 2)

@dataclass
class Config:
    """Application configuration settings."""
    input_mode: str = "folder"  # "folder" or "uvtools"

    # Folder mode settings
    input_folder: str = ""
    output_folder: str = ""

    # UVTools mode settings
    uvtools_path: str = ""
    uvtools_temp_folder: str = ""
    uvtools_input_file: str = ""
    output_file_prefix: str = "Dask_Processed_"
    uvtools_output_location: str = "working_folder" # "working_folder" or "input_folder"
    uvtools_delete_temp_on_completion: bool = True

    # Dask settings
    thread_count: int = DEFAULT_NUM_WORKERS
    chunk_size_x: int = 64
    chunk_size_y: int = 64
    chunk_size_z: int = 64

    def save(self, filepath: str):
        """Saves the current configuration to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Loads configuration from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create a new config instance with loaded data
        # This handles missing or extra keys gracefully
        config_instance = cls()
        for key, value in data.items():
            if hasattr(config_instance, key):
                setattr(config_instance, key, value)
        return config_instance

# Global application config instance
app_config = Config()
