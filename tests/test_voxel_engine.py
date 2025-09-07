# -*- coding: utf-8 -*-
"""
Module: test_voxel_engine.py
Author: Gemini
Description: Unit tests for the VoxelEngine.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

# Try to import pyopenvdb and skip tests if it's not available
try:
    import pyopenvdb as vdb
    PYOPENVDB_INSTALLED = True
except ImportError:
    PYOPENVDB_INSTALLED = False

from core.slice_loader import SliceLoader
from core.voxel_engine import VoxelEngine

# Pytest marker to skip tests if pyopenvdb is not installed
requires_openvdb = pytest.mark.skipif(not PYOPENVDB_INSTALLED, reason="pyopenvdb is not installed")

@pytest.fixture
def dummy_slice_dir(tmp_path):
    """Fixture to create a temporary directory with dummy slice files."""
    slice_dir = tmp_path / "slices"
    slice_dir.mkdir()

    num_files = 10
    shape = (50, 60)
    for i in range(num_files):
        filename = slice_dir / f"slice_{i:04d}.png"
        img = np.full(shape, i * 10, dtype=np.uint8) # Each slice has a different color
        cv2.imwrite(str(filename), img)

    return str(slice_dir)

@requires_openvdb
def test_voxel_engine_initialization(dummy_slice_dir):
    """
    Tests the initialization of the VoxelEngine.
    """
    # Arrange
    loader = SliceLoader(dummy_slice_dir)

    # Act
    engine = VoxelEngine(loader, window_size=5)

    # Assert
    assert engine.total_slices == 10
    assert engine.window_size == 5
    assert engine.width == 60
    assert engine.height == 50

@requires_openvdb
def test_voxel_engine_iteration(dummy_slice_dir):
    """
    Tests that the VoxelEngine correctly iterates and yields OpenVDB grids.
    """
    # Arrange
    loader = SliceLoader(dummy_slice_dir)
    engine = VoxelEngine(loader, window_size=3)

    # Act
    windows = list(engine.iter_windows())

    # Assert
    assert len(windows) == 10 - 3 + 1 # total_slices - window_size + 1

    # Check the first yielded grid
    first_grid = windows[0]
    assert isinstance(first_grid, vdb.FloatGrid)
    assert first_grid.is_class(vdb.GridClass.FOG_VOLUME)
    assert first_grid.active_voxel_count() > 0

@requires_openvdb
def test_voxel_engine_grid_content(dummy_slice_dir):
    """
    Tests the content of a yielded grid to ensure it matches the input.
    """
    # Arrange
    loader = SliceLoader(dummy_slice_dir)
    engine = VoxelEngine(loader, window_size=3)

    # Act
    first_grid = next(engine.iter_windows())

    # Assert
    # Convert grid back to a numpy array to check values
    # Note: This is an approximation. The background value in the dense array will be 0.
    dense_array = np.zeros((3, 50, 60), dtype=np.float32)
    first_grid.copy_to_dense(dense_array)

    # The values were 0, 10, 20, etc. Normalized to [0, 1].
    # We expect the first slice (z=0) of the grid to have values from the first image.
    expected_value_slice_0 = 0 / 255.0
    assert np.isclose(dense_array[0, 25, 30], expected_value_slice_0)

    # We expect the second slice (z=1) to have values from the second image.
    expected_value_slice_1 = 10 / 255.0
    assert np.isclose(dense_array[1, 25, 30], expected_value_slice_1)
