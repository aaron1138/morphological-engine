import json
import os
from typing import Any, Dict

def _deep_update(source: Dict, overrides: Dict) -> Dict:
    """
    Recursively update a dictionary.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = _deep_update(source[key], value)
        else:
            source[key] = value
    return source

class AppConfig:
    """
    Manages the application's configuration settings by loading from and
    saving to a JSON file.
    """
    def __init__(self, config_path: str = 'config.json'):
        """
        Initializes the AppConfig instance.

        Args:
            config_path (str): The path to the configuration file.
        """
        self.config_path = config_path
        self.settings = self._get_default_settings()
        self.load()

    def _get_default_settings(self) -> Dict[str, Any]:
        """
        Provides the default configuration dictionary.

        Returns:
            A dictionary with default settings.
        """
        return {
            'gpu_selection': {
                'device_id': 0,
                'auto_select': True
            },
            'ui': {
                'theme': 'dark',
                'window_size': [1280, 720]
            },
            'processing_presets': {}
        }

    def load(self) -> None:
        """
        Loads the configuration from the JSON file. If the file doesn't exist
        or is invalid, it creates one with default settings.
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                self.settings = _deep_update(self.settings, user_config)
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Could not decode {self.config_path}. Using default settings.")
                # If file is corrupt, we can choose to overwrite it with defaults
                self.save()
        else:
            self.save()

    def save(self) -> None:
        """Saves the current settings to the JSON file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except IOError as e:
            print(f"Error: Could not save config file to {self.config_path}. Details: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a setting using dot notation.

        Args:
            key (str): The key of the setting (e.g., 'gpu_selection.device_id').
            default (Any, optional): The value to return if the key is not found.

        Returns:
            The value of the setting or the default.
        """
        value = self.settings
        try:
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Sets a setting using dot notation and saves the configuration.

        Args:
            key (str): The key of the setting (e.g., 'ui.theme').
            value (Any): The new value to set.
        """
        keys = key.split('.')
        d = self.settings
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
        self.save()
