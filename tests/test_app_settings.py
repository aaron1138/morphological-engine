# -*- coding: utf-8 -*-
"""
Tests for the AppSettings manager.
"""

import os
import json
import pytest
from pathlib import Path

# Ensure the utils path is in the system path for importing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.app_settings import AppSettings

@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """A pytest fixture to create a temporary config file path."""
    return tmp_path / "test_config.json"

def test_init_creates_default_config(temp_config_file: Path):
    """Test that AppSettings creates a default config if one doesn't exist."""
    assert not temp_config_file.exists()
    settings = AppSettings(config_filename=str(temp_config_file))

    # Check that the file was created
    assert temp_config_file.exists()

    # Check that the content matches the default settings
    with open(temp_config_file, 'r') as f:
        data = json.load(f)

    assert data["gpu_device_id"] == -1
    assert data["ui_theme"] == "dark"

def test_load_existing_config(temp_config_file: Path):
    """Test loading an existing and valid configuration file."""
    # Create a custom config file first
    custom_config = {
        "gpu_device_id": 1,
        "ui_theme": "light",
        "last_slice_dir": "/custom/path"
    }
    with open(temp_config_file, 'w') as f:
        json.dump(custom_config, f)

    settings = AppSettings(config_filename=str(temp_config_file))

    # Check that the loaded settings match the custom config
    assert settings.get("gpu_device_id") == 1
    assert settings.get("ui_theme") == "light"
    assert settings.get("last_slice_dir") == "/custom/path"
    # Check that a default key not in the custom file is still present
    assert settings.get("save_debug_intermediates") is False

def test_get_and_set_values(temp_config_file: Path):
    """Test getting and setting values in the settings manager."""
    settings = AppSettings(config_filename=str(temp_config_file))

    # Test get
    assert settings.get("ui_theme") == "dark"

    # Test set
    settings.set("ui_theme", "blue")
    assert settings.get("ui_theme") == "blue"

    # Verify that the change was saved to the file
    with open(temp_config_file, 'r') as f:
        data = json.load(f)
    assert data["ui_theme"] == "blue"

def test_load_corrupt_config(temp_config_file: Path):
    """Test that loading a corrupt JSON file falls back to default settings."""
    # Write invalid JSON to the file
    with open(temp_config_file, 'w') as f:
        f.write("{'invalid_json': True,}")

    # AppSettings should handle the error and use defaults
    settings = AppSettings(config_filename=str(temp_config_file))

    # Check that settings are the default ones
    assert settings.get("gpu_device_id") == -1
    assert settings.get("ui_theme") == "dark"
