# -*- coding: utf-8 -*-
"""
Module: app_settings.py
Author: Jules
Description: Manages loading and saving of persistent, application-wide settings.
             This handles user preferences like GPU selection, UI state, etc.,
             and is distinct from the processing pipeline configurations.
"""

import json
from pathlib import Path
from typing import Dict, Any

class AppSettings:
    """
    A class to manage application settings, saved in a JSON file.
    """
    def __init__(self, config_filename: str = "app_config.json"):
        """
        Initializes the settings manager.

        Args:
            config_filename (str): The name of the configuration file. It will be
                                   stored in the user's home directory or a
                                   designated app data folder in a real app,
                                   but for simplicity, we'll place it in the
                                   project root for now.
        """
        self.config_path = Path(config_filename)
        self.settings = self._get_default_settings()
        self.load()

    def _get_default_settings(self) -> Dict[str, Any]:
        """Returns the default application settings."""
        return {
            "gpu_device_id": -1,  # -1 for auto/default
            "rawgl_executable_path": "", # Path to rawgl.exe
            "ui_theme": "dark",
            "last_slice_dir": None,
            "last_output_dir": None,
            "save_debug_intermediates": False
        }

    def load(self):
        """
        Loads settings from the JSON file. If the file doesn't exist,
        it saves the default settings to a new file.
        """
        if not self.config_path.exists():
            print(f"'{self.config_path}' not found. Creating with default settings.")
            self.save()
        else:
            try:
                with open(self.config_path, 'r') as f:
                    loaded_settings = json.load(f)
                # Merge loaded settings with defaults to ensure all keys are present
                self.settings.update(loaded_settings)
                print(f"Application settings loaded from '{self.config_path}'.")
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error loading '{self.config_path}': {e}. Using default settings.")
                # In case of a corrupt file, we fall back to defaults
                self.settings = self._get_default_settings()

    def save(self):
        """Saves the current settings to the JSON file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
            print(f"Application settings saved to '{self.config_path}'.")
        except IOError as e:
            print(f"Error saving settings to '{self.config_path}': {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a setting value by key."""
        return self.settings.get(key, default)

    def set(self, key: str, value: Any):
        """Sets a setting value by key and saves it."""
        self.settings[key] = value
        self.save()

# Example of how this might be used as a singleton instance across the app
# In __main__ or app startup:
# settings_manager = AppSettings()
#
# Elsewhere in the code:
# from utils.app_settings import settings_manager
# current_gpu = settings_manager.get("gpu_device_id")
# settings_manager.set("last_slice_dir", "/path/to/slices")
