import pytest
import json
import os

# Make the src directory available for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import AppConfig
from src.utils.logger import RunLogger

# --- Tests for AppConfig ---

def test_config_defaults(tmp_path):
    """Tests that default settings are loaded when no config file exists."""
    config_file = tmp_path / "config.json"
    config = AppConfig(config_path=str(config_file))

    assert config.get('ui.theme') == 'dark'
    assert config.get('gpu_selection.auto_select') is True
    # The file should be created with defaults upon initialization
    assert os.path.exists(config_file)
    with open(config_file, 'r') as f:
        data = json.load(f)
        assert data['ui']['theme'] == 'dark'

def test_config_set_and_get(tmp_path):
    """Tests setting and getting values, including nested ones."""
    config_file = tmp_path / "config.json"
    config = AppConfig(config_path=str(config_file))

    # Set a top-level value
    config.set('new_setting', 'new_value')
    assert config.get('new_setting') == 'new_value'

    # Set a nested value
    config.set('ui.theme', 'light')
    assert config.get('ui.theme') == 'light'

    # Set a deeper nested value that creates a new dict
    config.set('gpu_selection.advanced.setting', True)
    assert config.get('gpu_selection.advanced.setting') is True

def test_config_save_and_load(tmp_path):
    """Tests that settings are saved and loaded correctly across instances."""
    config_file = tmp_path / "config.json"

    # Create first instance, modify and save
    config1 = AppConfig(config_path=str(config_file))
    config1.set('ui.window_size', [1920, 1080])
    config1.set('processing_presets.default', {'param': 1})

    # Create a second instance to load the file
    config2 = AppConfig(config_path=str(config_file))
    assert config2.get('ui.window_size') == [1920, 1080]
    assert config2.get('processing_presets.default.param') == 1
    assert config2.get('ui.theme') == 'dark' # check default is still there

def test_config_load_corrupted_file(tmp_path):
    """Tests that defaults are used if the config file is corrupted."""
    config_file = tmp_path / "config.json"
    with open(config_file, 'w') as f:
        f.write("this is not json")

    config = AppConfig(config_path=str(config_file))
    # Should load defaults
    assert config.get('ui.theme') == 'dark'
    # And overwrite the corrupted file with defaults
    with open(config_file, 'r') as f:
        data = json.load(f)
        assert data['ui']['theme'] == 'dark'

# --- Tests for RunLogger ---

def test_logger_creates_file(tmp_path):
    """Tests that the logger creates a new log file."""
    log_file = tmp_path / "run_log.json"
    logger = RunLogger(log_path=str(log_file))

    assert not os.path.exists(log_file)
    logger.log_run({'input': 'file1.png', 'status': 'success'})
    assert os.path.exists(log_file)

def test_logger_appends_runs(tmp_path):
    """Tests that the logger appends new entries to an existing log."""
    log_file = tmp_path / "run_log.json"
    logger = RunLogger(log_path=str(log_file))

    # Log the first run
    run1_data = {'input': 'file1.png', 'shader': 'shader1'}
    logger.log_run(run1_data)

    # Log the second run
    run2_data = {'input': 'file2.png', 'shader': 'shader2'}
    logger.log_run(run2_data)

    # Read the file and verify
    with open(log_file, 'r') as f:
        logs = json.load(f)

    assert isinstance(logs, list)
    assert len(logs) == 2
    assert logs[0]['run_data'] == run1_data
    assert logs[1]['run_data'] == run2_data

def test_logger_handles_corrupted_file(tmp_path):
    """Tests that the logger starts a new log if the file is corrupted."""
    log_file = tmp_path / "run_log.json"
    with open(log_file, 'w') as f:
        f.write("this is not a json list")

    logger = RunLogger(log_path=str(log_file))
    run_data = {'input': 'file.png'}
    logger.log_run(run_data)

    with open(log_file, 'r') as f:
        logs = json.load(f)

    assert len(logs) == 1
    assert logs[0]['run_data'] == run_data
