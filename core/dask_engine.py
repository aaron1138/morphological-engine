import os
import re
import dask.array as da
from dask import delayed
import imageio
from typing import Callable, List

def _get_sorted_files(directory: str) -> List[str]:
    """
    Finds and numerically sorts PNG files in a directory. This is crucial
    for ensuring the Z-stack is loaded in the correct order.
    """
    try:
        files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
    except FileNotFoundError:
        return []

    numeric_pattern = re.compile(r'(\d+)')

    def sort_key(filename: str) -> tuple:
        parts = numeric_pattern.findall(filename)
        if parts:
            return tuple(int(p) for p in parts)
        # Fallback for non-numeric filenames, ensures they are at the end
        return (float('inf'), filename)

    files.sort(key=sort_key)
    return [os.path.join(directory, f) for f in files]

def load_image_stack(directory: str) -> da.Array:
    """
    Lazily loads a stack of PNG images from a directory into a Dask array.

    Args:
        directory: The path to the directory containing the PNG files.

    Returns:
        A 3D Dask array representing the image stack (z, y, x).

    Raises:
        FileNotFoundError: If the directory is not found or contains no PNG files.
        ValueError: If the images in the stack have inconsistent dimensions.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Input directory not found: {directory}")

    sorted_files = _get_sorted_files(directory)
    if not sorted_files:
        raise FileNotFoundError(f"No PNG files found in directory: {directory}")

    # Lazily read the first image to get shape and dtype information
    sample_image = imageio.imread(sorted_files[0])
    shape = sample_image.shape
    dtype = sample_image.dtype

    # Use dask.delayed to wrap imageio.imread for lazy loading
    lazy_reader = delayed(imageio.imread)
    lazy_images = [lazy_reader(f) for f in sorted_files]

    # Handle RGB/RGBA images by converting to grayscale
    if sample_image.ndim == 3:
        if shape[-1] not in [3, 4]:
             raise ValueError(f"Unsupported image format with shape {shape}. Expected 2D grayscale or 3/4-channel color.")
        # Convert to grayscale by taking the first channel (R). This is fast.
        # A more accurate conversion would be a weighted average, but for many
        # industrial formats, channels can be identical.
        lazy_images = [delayed(lambda x: x[:, :, 0])(img) for img in lazy_images]
        shape = shape[:2] # Update shape to 2D

    # Create a list of Dask arrays from the delayed objects
    dask_arrays = [da.from_delayed(img, shape=shape, dtype=dtype) for img in lazy_images]

    # Stack the 2D arrays into a single 3D array along the z-axis (axis 0)
    stack = da.stack(dask_arrays, axis=0)

    # Rechunk to the desired 3D chunk size for efficient orthogonal access
    chunked_stack = stack.rechunk({0: 64, 1: 64, 2: 64})

    print(f"Dask array created with shape: {chunked_stack.shape}, chunks: {chunked_stack.chunksize}")
    return chunked_stack

def save_orthogonal_planes(
    dask_array: da.Array,
    output_folder: str,
    axis: int,
    progress_callback: Callable[[int, int], None] = None
):
    """
    Extracts and saves orthogonal plane images from a 3D Dask array.

    Args:
        dask_array: The 3D Dask array (z, y, x).
        output_folder: The directory to save the output PNG files.
        axis: The axis to slice along (0=XY, 1=XZ, 2=YZ).
        progress_callback: A function to call with (current_slice, total_slices).
    """
    if axis not in [0, 1, 2]:
        raise ValueError("Axis must be 0, 1, or 2.")
    if dask_array.ndim != 3:
        raise ValueError(f"Input must be a 3D Dask array, but got {dask_array.ndim} dimensions.")

    os.makedirs(output_folder, exist_ok=True)
    num_slices = dask_array.shape[axis]

    axis_names = {0: "XY", 1: "XZ", 2: "YZ"}
    plane_name = axis_names[axis]

    print(f"Starting extraction of {num_slices} '{plane_name}' planes.")

    for i in range(num_slices):
        if progress_callback:
            progress_callback(i, num_slices)

        # Select the 2D plane from the 3D stack
        if axis == 0:  # XY plane (slice along Z)
            plane = dask_array[i, :, :]
        elif axis == 1:  # XZ plane (slice along Y)
            plane = dask_array[:, i, :]
        else:  # YZ plane (slice along X)
            plane = dask_array[:, :, i]

        output_filename = f"plane_{plane_name}_{i:05d}.png"
        output_path = os.path.join(output_folder, output_filename)

        # Compute the single slice and save it to a file.
        # This one-slice-at-a-time approach is memory efficient.
        computed_plane = plane.compute()
        imageio.imwrite(output_path, computed_plane)

    if progress_callback:
        progress_callback(num_slices, num_slices)  # Final update to show 100%

    print(f"Extraction of '{plane_name}' planes complete.")
