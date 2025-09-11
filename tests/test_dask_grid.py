import pytest
import numpy as np
import imageio
from pathlib import Path
import sys
import os

# Make the src directory available for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.dask_grid import DaskGrid

@pytest.fixture(scope="module")
def image_stack_directory(tmp_path_factory):
    """Creates a temporary directory with a stack of dummy PNG images."""
    tmp_dir = tmp_path_factory.mktemp("image_stack")
    # Create 5 images, 10x20 pixels
    for i in range(5):
        # Create a simple gradient image
        img_data = np.full((10, 20), i * 10, dtype=np.uint8)
        imageio.imwrite(tmp_dir / f"{i}.png", img_data)
    return tmp_dir

def test_dask_grid_initialization(image_stack_directory):
    """Tests that the DaskGrid initializes correctly."""
    grid = DaskGrid(str(image_stack_directory))
    # Shape should be (num_images, height, width)
    assert grid.shape == (5, 10, 20)
    assert grid.dask_array is not None

def test_dask_grid_get_xy_slice(image_stack_directory):
    """Tests getting an XY slice (an original image)."""
    grid = DaskGrid(str(image_stack_directory))
    xy_slice = grid.get_slice('XY', 2)
    assert xy_slice.shape == (10, 20)

    # Compute the slice and check its content
    computed_slice = xy_slice.compute()
    # The 3rd image (index 2) should have a value of 2*10=20
    assert np.all(computed_slice == 20)

def test_dask_grid_get_xz_slice(image_stack_directory):
    """Tests getting an orthogonal XZ slice."""
    grid = DaskGrid(str(image_stack_directory))
    # Get a slice along the Y axis, at y-index 3
    xz_slice = grid.get_slice('XZ', 3)
    assert xz_slice.shape == (5, 20) # (depth, width)

    # Compute the slice and check its content
    computed_slice = xz_slice.compute()
    # The values should be [0, 10, 20, 30, 40] repeated across the width
    expected_column = np.arange(0, 50, 10)
    assert np.all(computed_slice == expected_column[:, np.newaxis])

def test_dask_grid_get_yz_slice(image_stack_directory):
    """Tests getting an orthogonal YZ slice."""
    grid = DaskGrid(str(image_stack_directory))
    # Get a slice along the X axis, at x-index 5
    yz_slice = grid.get_slice('YZ', 5)
    assert yz_slice.shape == (5, 10) # (depth, height)

    # Compute the slice and check its content
    computed_slice = yz_slice.compute()
    expected_column = np.arange(0, 50, 10)
    assert np.all(computed_slice == expected_column[:, np.newaxis])

def test_dask_grid_invalid_plane(image_stack_directory):
    """Tests that an invalid plane name raises an error."""
    grid = DaskGrid(str(image_stack_directory))
    with pytest.raises(ValueError):
        grid.get_slice('XX', 0)
