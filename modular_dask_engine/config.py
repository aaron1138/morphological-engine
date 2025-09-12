import os
import json
from dataclasses import dataclass, field, asdict

@dataclass
class Config:
    """Main application configuration."""
    # --- I/O Settings ---
    input_mode: str = "folder"  # "folder" or "uvtools"
    input_folder: str = ""
    output_folder: str = ""

    # --- UVTools Mode Settings ---
    uvtools_path: str = "C:\\Program Files\\UVTools\\UVToolsCmd.exe"
    uvtools_temp_folder: str = ""
    uvtools_input_file: str = ""
    uvtools_output_location: str = "working_folder"
    output_file_prefix: str = "Dask_Processed_"

    # --- Dask Settings ---
    thread_count: int = 4
    chunk_size_x: int = 64
    chunk_size_y: int = 64
    chunk_size_z: int = 64

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
            return cls()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            print(f"Error loading config '{filepath}': {e}. Using default.")
            return cls()

app_config = Config.load("app_config.json")
