import dask
import dask.array as da
import imageio.v2 as imageio
import os
import re
from typing import List
import numpy as np

# In the future, Numba can be used to accelerate custom, per-pixel or per-chunk
# operations in a filter pipeline. For example:
#
# from numba import jit
#
# @jit(nopython=True)
# def numba_accelerated_filter(chunk):
#     # ... implementation ...
#

def _get_sorted_files(folder: str) -> List[str]:
    """Gets a numerically sorted list of PNG files in a folder."""
    numeric_pattern = re.compile(r'(\d+)\.\w+$')
    def get_numeric_part(filename):
        match = numeric_pattern.search(filename)
        return int(match.group(1)) if match else float('inf')

    return sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.png')],
        key=lambda f: get_numeric_part(os.path.basename(f))
    )

def load_image_stack(input_folder: str) -> da.Array:
    """
    Reads a folder of PNGs into a Dask array.
    The images are stacked along the first axis (Z).
    The array is chunked to (64, 64, 64).
    """
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder not found: {input_folder}")

    image_files = _get_sorted_files(input_folder)
    if not image_files:
        raise ValueError(f"No PNG files found in {input_folder}")

    # Read the first image to get shape and dtype
    sample_image = imageio.imread(image_files[0])
    shape = sample_image.shape
    dtype = sample_image.dtype

    # Create a list of delayed image reads
    lazy_images = [dask.delayed(imageio.imread)(f) for f in image_files]

    # Create a Dask array from the delayed images
    arrays = [da.from_delayed(img, shape=shape, dtype=dtype) for img in lazy_images]
    stack = da.stack(arrays, axis=0)

    # Rechunk the array to the desired chunk size
    # Note: If images are RGB, shape will be (z, y, x, c). We only chunk z, y, x.
    chunk_sizes = {0: 64, 1: 64, 2: 64}
    if len(shape) > 2: # Handle color channels if they exist
        chunk_sizes[3] = shape[2]

    chunked_stack = stack.rechunk(chunk_sizes)

    return chunked_stack

def save_planes(dask_array: da.Array, output_folder: str, plane: str):
    """
    Saves orthogonal planes from a Dask array to PNG files.
    plane: 'xy', 'xz', or 'yz'
    """
    output_plane_folder = os.path.join(output_folder, plane.lower() + "_planes")
    if not os.path.exists(output_plane_folder):
        os.makedirs(output_plane_folder)

    if plane.lower() == 'xy':
        # XY planes are the original Z-slices
        for i in range(dask_array.shape[0]):
            slice_data = dask_array[i, :, :].compute()
            imageio.imwrite(os.path.join(output_plane_folder, f"xy_plane_{i:04d}.png"), slice_data)

    elif plane.lower() == 'xz':
        # XZ planes are slices along the Y-axis
        for i in range(dask_array.shape[1]):
            slice_data = dask_array[:, i, :].compute()
            imageio.imwrite(os.path.join(output_plane_folder, f"xz_plane_{i:04d}.png"), slice_data)

    elif plane.lower() == 'yz':
        # YZ planes are slices along the X-axis
        for i in range(dask_array.shape[2]):
            slice_data = dask_array[:, :, i].compute()
            imageio.imwrite(os.path.join(output_plane_folder, f"yz_plane_{i:04d}.png"), slice_data)
    else:
        raise ValueError("Invalid plane type. Must be 'xy', 'xz', or 'yz'.")
