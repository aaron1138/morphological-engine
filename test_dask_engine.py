import os
import shutil
import numpy as np
import pytest
import imageio.v2 as imageio
import dask.array as da
from dask_engine import load_image_stack, save_planes

@pytest.fixture
def image_stack_folder():
    """Creates a temporary folder with a stack of dummy PNG images."""
    folder_path = "test_image_stack"
    os.makedirs(folder_path, exist_ok=True)

    # Create 10 dummy images of size 128x128
    for i in range(10):
        img = np.zeros((128, 128), dtype=np.uint8)
        img[:, :64] = i * 10  # Some variation
        imageio.imwrite(os.path.join(folder_path, f"slice_{i:04d}.png"), img)

    yield folder_path

    # Cleanup
    shutil.rmtree(folder_path)

def test_load_image_stack(image_stack_folder):
    """Tests loading an image stack into a Dask array."""
    dask_array = load_image_stack(image_stack_folder)

    assert isinstance(dask_array, da.Array)
    assert dask_array.shape == (10, 128, 128)
    assert dask_array.chunksize == (10, 64, 64) # Z-chunk is 10 because there are only 10 images

def test_save_planes(image_stack_folder):
    """Tests saving orthogonal planes."""
    output_folder = "test_output_planes"
    os.makedirs(output_folder, exist_ok=True)

    dask_array = load_image_stack(image_stack_folder)

    # Test XY plane extraction
    save_planes(dask_array, output_folder, "xy")
    xy_output_path = os.path.join(output_folder, "xy_planes")
    assert os.path.isdir(xy_output_path)
    assert len(os.listdir(xy_output_path)) == 10 # 10 XY planes

    # Test XZ plane extraction
    save_planes(dask_array, output_folder, "xz")
    xz_output_path = os.path.join(output_folder, "xz_planes")
    assert os.path.isdir(xz_output_path)
    assert len(os.listdir(xz_output_path)) == 128 # 128 XZ planes

    # Test YZ plane extraction
    save_planes(dask_array, output_folder, "yz")
    yz_output_path = os.path.join(output_folder, "yz_planes")
    assert os.path.isdir(yz_output_path)
    assert len(os.listdir(yz_output_path)) == 128 # 128 YZ planes

    # Cleanup
    shutil.rmtree(output_folder)
