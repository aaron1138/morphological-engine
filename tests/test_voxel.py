import pytest
import numpy as np
import openvdb

# Make the src directory available for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.voxel import VoxelGrid

@pytest.fixture
def sample_numpy_array() -> np.ndarray:
    """Provides a sample 3D numpy array for testing."""
    arr = np.zeros((5, 5, 5), dtype=float)
    arr[2, 2, 2] = 255.0
    arr[1, 2, 3] = 128.0
    return arr

def test_voxel_grid_initialization():
    """Tests the creation of an empty VoxelGrid."""
    grid = VoxelGrid()
    assert isinstance(grid.grid, openvdb.FloatGrid)
    assert not grid.grid.has_active_voxels()

def test_from_and_to_numpy(sample_numpy_array):
    """Tests converting from and to a NumPy array."""
    voxel_grid = VoxelGrid.from_numpy(sample_numpy_array)
    assert voxel_grid.grid.has_active_voxels()

    # Convert back to numpy
    numpy_out, offset = voxel_grid.to_numpy()

    # Check that the dimensions and content match
    assert np.array_equal(sample_numpy_array, numpy_out)
    assert offset == (0, 0, 0)

def test_set_and_get_voxel():
    """Tests setting and getting individual voxel values."""
    voxel_grid = VoxelGrid()

    # Set a voxel value
    voxel_grid.set_voxel(10, 20, 30, 128.0)

    # Get the value back and assert it's correct
    value = voxel_grid.get_voxel(10, 20, 30)
    assert value == 128.0

    # Check a different, unset voxel
    background_value = voxel_grid.get_voxel(5, 5, 5)
    assert background_value == 0.0 # Default background value

def test_save_and_load_from_file(tmp_path, sample_numpy_array):
    """Tests saving a grid to a file and loading it back."""
    # Create a file path in the temporary directory
    file_path = tmp_path / "test_grid.vdb"

    # Create a grid and save it
    original_grid = VoxelGrid.from_numpy(sample_numpy_array)
    original_grid.save(str(file_path))

    # Load the grid from the file
    loaded_grid = VoxelGrid.from_file(str(file_path))

    # Verify the loaded grid is the same as the original
    original_np, _ = original_grid.to_numpy()
    loaded_np, _ = loaded_grid.to_numpy()

    assert np.array_equal(original_np, loaded_np)

def test_load_nonexistent_file():
    """Tests that loading a non-existent file raises an error."""
    with pytest.raises((IOError, RuntimeError)):
        VoxelGrid.from_file("nonexistent_file.vdb")
