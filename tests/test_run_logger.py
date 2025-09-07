# -*- coding: utf-8 -*-
"""
Module: test_run_logger.py
Author: Gemini
Description: Unit tests for the RunLogger utility.
"""

import os
import json
import pytest
from utils.run_logger import RunLogger

@pytest.fixture
def temp_log_file(tmp_path):
    """Fixture to provide a temporary log file path."""
    return tmp_path / "test_run_log.json"

def test_log_creation(temp_log_file):
    """
    Tests that a log file is created on the first log write.
    """
    # Arrange
    logger = RunLogger(log_file_path=str(temp_log_file))
    run_data = {"status": "test"}

    # Assert Pre-condition
    assert not os.path.exists(temp_log_file)

    # Act
    logger.log_run(run_data)

    # Assert
    assert os.path.exists(temp_log_file)

def test_log_multiple_runs(temp_log_file):
    """
    Tests that multiple runs are appended correctly to the log file.
    """
    # Arrange
    logger = RunLogger(log_file_path=str(temp_log_file))
    run1_data = {"id": 1, "status": "Completed"}
    run2_data = {"id": 2, "status": "Failed"}

    # Act
    logger.log_run(run1_data)
    logger.log_run(run2_data)

    # Assert
    with open(temp_log_file, 'r') as f:
        log_content = json.load(f)

    assert isinstance(log_content, list)
    assert len(log_content) == 2
    assert log_content[0]["run_details"]["id"] == 1
    assert log_content[1]["run_details"]["id"] == 2
    assert "timestamp" in log_content[0]

def test_log_read_nonexistent(temp_log_file):
    """
    Tests that reading a non-existent log returns an empty list.
    """
    # Arrange
    logger = RunLogger(log_file_path=str(temp_log_file))

    # Act
    logs = logger._read_log()

    # Assert
    assert logs == []

def test_log_read_corrupted(temp_log_file):
    """
    Tests that reading a corrupted log file returns an empty list.
    """
    # Arrange
    with open(temp_log_file, 'w') as f:
        f.write("this is not json")

    logger = RunLogger(log_file_path=str(temp_log_file))

    # Act
    logs = logger._read_log()

    # Assert
    assert logs == []
