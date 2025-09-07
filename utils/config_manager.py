# -*- coding: utf-8 -*-
"""
Module: config_manager.py
Author: Gemini
Description: Handles saving/loading of processing pipeline configurations
             and manages the application's persistent settings.
"""

import json
from typing import Dict, Any

# --- App-specific settings management ---

APP_CONFIG_FILE = 'app_config.json'

def get_default_app_config() -> Dict[str, Any]:
    """Returns the default application settings."""
    return {
        "last_slice_directory": "",
        "last_output_directory": "",
        "selected_gpu": None,
        "save_debug_steps": False,
        "window_geometry": {
            "x": 100,
            "y": 100,
            "width": 1280,
            "height": 720
        }
    }

def load_app_config() -> Dict[str, Any]:
    """
    Loads the application's configuration from app_config.json.
    If the file doesn't exist or is invalid, it returns the default config.
    """
    try:
        with open(APP_CONFIG_FILE, 'r') as f:
            config = json.load(f)
        # Ensure all default keys are present
        defaults = get_default_app_config()
        for key, value in defaults.items():
            config.setdefault(key, value)
        return config
    except (FileNotFoundError, json.JSONDecodeError):
        return get_default_app_config()

def save_app_config(config: Dict[str, Any]):
    """Saves the application's configuration to app_config.json."""
    try:
        with open(APP_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except IOError as e:
        print(f"Error saving app configuration: {e}")


# --- Processing pipeline configuration management (for user save/load) ---

def save_configuration(config: Dict[str, Any], file_path: str):
    """
    Saves a configuration dictionary to a JSON file.

    Args:
        config (Dict[str, Any]): The configuration dictionary to save.
        file_path (str): The full path to the output file.

    Raises:
        IOError: If there is an error writing the file.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration successfully saved to {file_path}")
    except IOError as e:
        print(f"Error saving configuration file: {e}")
        raise

def load_configuration(file_path: str) -> Dict[str, Any]:
    """
    Loads a configuration dictionary from a JSON file.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary.

    Raises:
        IOError: If there is an error reading the file.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        print(f"Configuration successfully loaded from {file_path}")
        return config
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading configuration file: {e}")
        raise
