import os
import shutil
import numpy as np
import imageio
import pytest
from src.engine.dask_engine import DaskEngine

@pytest.fixture
def image_stack_folder():
    """Creates a temporary folder with a stack of PNG images for testing."""
    temp_dir = "temp_test_images"
    os.makedirs(temp_dir, exist_ok=True)

    # Create 10 dummy images of size 128x128
    for i in range(10):
        img = np.random.randint(0, 256, size=(128, 128), dtype=np.uint8)
        imageio.imwrite(os.path.join(temp_dir, f"image_{i}.png"), img)

    yield temp_dir

    # Clean up the temporary folder
    shutil.rmtree(temp_dir)

def test_load_image_stack(image_stack_folder):
    """Tests loading an image stack with the DaskEngine."""
    engine = DaskEngine()
    stack = engine.load_image_stack(
        input_folder=image_stack_folder,
        chunks=(5, 64, 64) # Z, Y, X
    )

    assert stack.shape == (10, 128, 128)
    assert stack.chunksize == (5, 64, 64)

    # Test getting planes
    xy_plane = engine.get_xy_plane(0)
    assert xy_plane.shape == (128, 128)

    xz_plane = engine.get_xz_plane(0)
    assert xz_plane.shape == (10, 128)

    yz_plane = engine.get_yz_plane(0)
    assert yz_plane.shape == (10, 128)

def test_load_rgb_image_stack():
    """Tests loading an RGB image stack and converting to grayscale."""
    temp_dir = "temp_test_rgb_images"
    os.makedirs(temp_dir, exist_ok=True)

    # Create 5 dummy RGB images
    for i in range(5):
        img = np.random.randint(0, 256, size=(100, 150, 3), dtype=np.uint8)
        imageio.imwrite(os.path.join(temp_dir, f"rgb_image_{i}.png"), img)

    engine = DaskEngine()
    stack = engine.load_image_stack(
        input_folder=temp_dir,
        chunks=(2, 50, 75)
    )

    assert stack.shape == (5, 100, 150)
    assert stack.chunksize == (2, 50, 75)

    # Clean up
    shutil.rmtree(temp_dir)
