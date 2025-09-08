# -*- coding: utf-8 -*-
"""
Tests for the run_logger utility.
"""

import os
import json
import pytest
from pathlib import Path

# Ensure the utils path is in the system path for importing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.run_logger import log_run

@pytest.fixture
def temp_log_file(tmp_path: Path) -> Path:
    """A pytest fixture to create a temporary log file path."""
    return tmp_path / "test_runs.jsonl"

def test_log_creates_file_and_adds_entry(temp_log_file: Path):
    """Test that logging creates a new file and adds a single entry."""
    assert not temp_log_file.exists()

    run_info = {"test_id": 1, "data": "some_data"}
    log_run(run_info, log_filename=str(temp_log_file))

    assert temp_log_file.exists()

    with open(temp_log_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 1

    log_entry = json.loads(lines[0])
    assert log_entry["test_id"] == 1
    assert log_entry["data"] == "some_data"
    # Check that a timestamp was added automatically
    assert "timestamp" in log_entry

def test_log_appends_to_existing_file(temp_log_file: Path):
    """Test that subsequent logs append to the same file."""
    # First entry
    run_info_1 = {"test_id": 1}
    log_run(run_info_1, log_filename=str(temp_log_file))

    # Second entry
    run_info_2 = {"test_id": 2}
    log_run(run_info_2, log_filename=str(temp_log_file))

    with open(temp_log_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 2

    entry_1 = json.loads(lines[0])
    entry_2 = json.loads(lines[1])

    assert entry_1["test_id"] == 1
    assert entry_2["test_id"] == 2

def test_log_uses_provided_timestamp(temp_log_file: Path):
    """Test that a provided timestamp is used instead of auto-generating one."""
    custom_timestamp = "2025-01-01T12:00:00Z"
    run_info = {"test_id": 3, "timestamp": custom_timestamp}
    log_run(run_info, log_filename=str(temp_log_file))

    with open(temp_log_file, 'r') as f:
        line = f.readline()

    log_entry = json.loads(line)
    assert log_entry["timestamp"] == custom_timestamp
