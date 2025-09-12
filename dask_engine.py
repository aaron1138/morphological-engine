# dask_engine.py
import os
import re
import dask.array as da
from dask import delayed
import imageio.v2 as imageio
import numpy as np
from numba import jit

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """
    Key for natural sorting of strings.
    """
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def load_image_stack(folder_path, chunk_size=(64, 64, 64)):
    """
    Lazily loads a stack of PNG images from a folder into a Dask array.

    Args:
        folder_path (str): The path to the folder containing the PNG images.
        chunk_size (tuple): The desired chunk size for the Dask array.

    Returns:
        dask.array: A 3D Dask array representing the image stack.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Input path is not a directory: {folder_path}")

    image_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.png')],
        key=natural_sort_key
    )

    if not image_files:
        raise ValueError(f"No PNG files found in directory: {folder_path}")

    # Get the shape of the first image to determine the shape for the whole stack
    try:
        sample_image = imageio.imread(image_files[0])
        shape = sample_image.shape
        dtype = sample_image.dtype
    except Exception as e:
        raise IOError(f"Could not read the first image to determine stack properties: {e}")

    # Create a list of delayed objects for reading each image
    lazy_imread = delayed(imageio.imread)
    lazy_images = [lazy_imread(f) for f in image_files]

    # Create a Dask array from the delayed images
    stack = [da.from_delayed(lazy_image, shape=shape, dtype=dtype) for lazy_image in lazy_images]

    # Stack the images along the Z-axis (axis 0)
    dask_stack = da.stack(stack, axis=0)

    # Rechunk the array to the desired chunk size
    # This is important for efficient slicing in all directions.
    dask_stack = dask_stack.rechunk(chunk_size)

    return dask_stack

def get_xy_plane(dask_array, z_index):
    """
    Extracts an XY plane from the 3D Dask array.

    Args:
        dask_array (dask.array): The 3D Dask array.
        z_index (int): The index of the plane to extract along the Z-axis.

    Returns:
        dask.array: A 2D Dask array representing the XY plane.
    """
    return dask_array[z_index, :, :]

def get_xz_plane(dask_array, y_index):
    """
    Extracts an XZ plane from the 3D Dask array.

    Args:
        dask_array (dask.array): The 3D Dask array.
        y_index (int): The index of the plane to extract along the Y-axis.

    Returns:
        dask.array: A 2D Dask array representing the XZ plane.
    """
    return dask_array[:, y_index, :]

def get_yz_plane(dask_array, x_index):
    """
    Extracts a YZ plane from the 3D Dask array.

    Args:
        dask_array (dask.array): The 3D Dask array.
        x_index (int): The index of the plane to extract along the X-axis.

    Returns:
        dask.array: A 2D Dask array representing the YZ plane.
    """
    return dask_array[:, :, x_index]

@jit(nopython=True, cache=True)
def example_numba_filter(image_plane):
    """
    An example of a Numba-accelerated filter that could be applied to an image plane.
    This function is a placeholder to demonstrate Numba integration.

    Args:
        image_plane (np.array): A 2D NumPy array representing an image plane.

    Returns:
        np.array: The processed 2D NumPy array.
    """
    # This is a simple threshold filter. In a real application, this could be
    # a more complex operation like a convolution, custom blur, etc.
    output_plane = np.zeros_like(image_plane)
    for y in range(image_plane.shape[0]):
        for x in range(image_plane.shape[1]):
            if image_plane[y, x] > 128:
                output_plane[y, x] = 255
            else:
                output_plane[y, x] = 0
    return output_plane

# Future filter pipelines would use these functions. For example, a user might
# want to apply a series of filters to each XZ plane. The engine is now set up
# to extract these planes efficiently.
#
# Example usage (for demonstration, not to be run directly here):
#
# def apply_filter_to_xz_planes(dask_stack, output_folder):
#     for y in range(dask_stack.shape[1]):
#         # Extract the plane
#         xz_plane = get_xz_plane(dask_stack, y).compute() # .compute() gets the NumPy array
#
#         # Apply a Numba-accelerated filter
#         filtered_plane = example_numba_filter(xz_plane)
#
#         # Save the result
#         imageio.imwrite(os.path.join(output_folder, f"xz_plane_{y:04d}.png"), filtered_plane)
#
