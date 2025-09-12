import os
import json
from dataclasses import dataclass, asdict

# A sensible default for worker threads, leaving some cores for the OS
DEFAULT_DASK_WORKERS = max(1, os.cpu_count() - 2 if os.cpu_count() else 1)

@dataclass
class Config:
    """
    Main application configuration for the Dask-based image processing engine.
    """
    # --- I/O Settings ---
    input_mode: str = "folder"  # "folder" or "uvtools"
    input_folder: str = ""
    output_folder: str = ""

    # --- UVTools Mode Settings ---
    uvtools_path: str = "C:\\Program Files\\UVTools\\UVToolsCmd.exe"
    uvtools_temp_folder: str = ""
    uvtools_input_file: str = ""
    uvtools_delete_temp_on_completion: bool = True
    output_file_prefix: str = "Dask_Engine_Processed_"

    # --- Dask Settings ---
    dask_workers: int = DEFAULT_DASK_WORKERS

    # --- Numba Settings ---
    use_numba: bool = True

    def to_dict(self) -> dict:
        """Serializes the configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Creates a Config instance from a dictionary, ignoring unknown keys."""
        config_instance = cls()
        for key, value in data.items():
            if hasattr(config_instance, key):
                # You might want to add type checking here for more robustness
                setattr(config_instance, key, value)
        return config_instance

    def save(self, filepath: str):
        """Saves the current configuration to a JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=4)
        except IOError as e:
            print(f"Error saving configuration to {filepath}: {e}")
            raise

    @classmethod
    def load(cls, filepath: str) -> "Config":
        """Loads configuration from a JSON file, or creates a default one."""
        if not os.path.exists(filepath):
            default_config = cls()
            try:
                # Save a default config file if it doesn't exist
                default_config.save(filepath)
            except IOError as e:
                print(f"Warning: Could not save default config to '{filepath}': {e}")
            return default_config

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error loading config file '{filepath}': {e}. Returning default config.")
            return cls()

# Global config instance, loaded from a default file.
# This can be updated by the GUI and saved/loaded to other files.
APP_CONFIG_FILE = "dask_engine_config.json"
app_config = Config.load(APP_CONFIG_FILE)
