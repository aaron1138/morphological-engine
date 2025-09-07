# -*- coding: utf-8 -*-
"""
Module: test_config_manager.py
Author: Gemini
Description: Unit tests for the application settings management in config_manager.
"""

import os
import json
import pytest
from utils import config_manager

@pytest.fixture
def temp_config_env(tmp_path):
    """Fixture to run tests in a temporary directory to isolate config files."""
    # Change the current working directory to the temporary directory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    # Yield control to the test function
    yield

    # Teardown: change back to the original directory
    os.chdir(original_cwd)


def test_load_default_config(temp_config_env):
    """
    Tests that load_app_config returns the default configuration when
    the config file does not exist.
    """
    # Act
    config = config_manager.load_app_config()

    # Assert
    default_config = config_manager.get_default_app_config()
    assert config == default_config
    assert not os.path.exists(config_manager.APP_CONFIG_FILE)

def test_save_and_load_config(temp_config_env):
    """
    Tests that saving and then loading a configuration preserves the data.
    """
    # Arrange
    test_config = config_manager.get_default_app_config()
    test_config["last_slice_directory"] = "C:/test/slices"
    test_config["window_geometry"]["width"] = 1600

    # Act
    config_manager.save_app_config(test_config)

    # Assert Pre-condition
    assert os.path.exists(config_manager.APP_CONFIG_FILE)

    # Act again
    loaded_config = config_manager.load_app_config()

    # Assert
    assert loaded_config == test_config
    assert loaded_config["last_slice_directory"] == "C:/test/slices"
    assert loaded_config["window_geometry"]["width"] == 1600

def test_load_config_with_missing_keys(temp_config_env):
    """
    Tests that loading a partially complete config file results in a config
    with all default keys present.
    """
    # Arrange
    partial_config = {
        "last_slice_directory": "/some/path",
        "selected_gpu": "Test GPU"
    }
    with open(config_manager.APP_CONFIG_FILE, 'w') as f:
        json.dump(partial_config, f)

    # Act
    loaded_config = config_manager.load_app_config()

    # Assert
    default_config = config_manager.get_default_app_config()
    assert loaded_config["last_slice_directory"] == "/some/path" # Should keep existing value
    assert loaded_config["selected_gpu"] == "Test GPU"
    assert "window_geometry" in loaded_config # Should have added missing key
    assert loaded_config["window_geometry"] == default_config["window_geometry"]
