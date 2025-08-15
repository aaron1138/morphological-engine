# -*- coding: utf-8 -*-
"""
Module: config_manager.py
Author: Gemini
Description: Handles saving and loading of processing pipeline configurations
             to and from JSON files.
"""

import json
from typing import Dict, Any

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
