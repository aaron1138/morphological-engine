# core/engine.py

import os
import re
import dask
import dask.array as da
import numpy as np
import imageio.v2 as imageio
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import numba
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@numba.jit(nopython=True)
def process_xy_slice(slice_2d):
    """Placeholder for a Numba-accelerated operation on an XY slice."""
    # Example operation: Invert the image
    return 255 - slice_2d

@numba.jit(nopython=True)
def process_xz_slice(slice_2d):
    """Placeholder for a Numba-accelerated operation on an XZ slice."""
    return slice_2d * 1.0 # No-op

@numba.jit(nopython=True)
def process_yz_slice(slice_2d):
    """Placeholder for a Numba-accelerated operation on a YZ slice."""
    return slice_2d * 1.0 # No-op

def _read_and_prepare_image(path):
    """Reads an image and converts it to 8-bit grayscale if necessary."""
    try:
        img = imageio.imread(path)
        if img.ndim == 3:
            # Convert RGB to grayscale
            img = img_as_ubyte(rgb2gray(img))
        elif img.dtype != np.uint8:
            # Ensure image is 8-bit
            img = img.astype(np.uint8)
        return img
    except Exception as e:
        logging.error(f"Could not read or process image {path}: {e}")
        return None


def create_dask_stack(image_folder: str) -> da.Array:
    """
    Creates a 3D Dask array from a folder of images, representing a Z-stack.

    Args:
        image_folder: The path to the folder containing numbered PNG images.

    Returns:
        A 3D Dask array with dimensions (Z, Y, X), chunked in (64, 64, 64).

    Raises:
        FileNotFoundError: If the image_folder does not exist.
        ValueError: If no valid images are found in the folder.
    """
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"Input directory not found: {image_folder}")

    # Robustly find and sort image files based on numbers in the filename
    numeric_pattern = re.compile(r'(\d+)')
    def get_numeric_part(filename):
        parts = numeric_pattern.findall(filename)
        return int(''.join(parts)) if parts else float('inf')

    try:
        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.lower().endswith('.png')],
            key=get_numeric_part
        )
    except Exception as e:
        logging.error(f"Could not sort files in {image_folder}: {e}")
        image_files = []

    if not image_files:
        raise ValueError(f"No PNG images found in '{image_folder}'")

    image_paths = [os.path.join(image_folder, f) for f in image_files]

    # Read the first image to determine shape and dtype for the stack
    first_image = _read_and_prepare_image(image_paths[0])
    if first_image is None:
        raise ValueError(f"Could not read the first image: {image_paths[0]}")

    img_shape = first_image.shape
    img_dtype = first_image.dtype
    logging.info(f"Detected image properties: Shape={img_shape}, Dtype={img_dtype}")

    # Create a list of delayed read operations
    lazy_imread = dask.delayed(_read_and_prepare_image)
    lazy_images = [lazy_imread(path) for path in image_paths]

    # Create the Dask array from the stack of delayed images
    # The final array shape will be (num_images, height, width)
    stack_shape = (len(lazy_images),) + img_shape

    dask_stack = da.from_delayed(
        lazy_images, shape=stack_shape, dtype=img_dtype
    )

    # Rechunk the array to the desired 3D chunk size
    # Dask handles the rechunking operation efficiently.
    chunk_size = (64, 64, 64)
    dask_stack = dask_stack.rechunk(chunk_size)

    logging.info(f"Created Dask array with shape {dask_stack.shape} and chunks {dask_stack.chunksize}")

    return dask_stack
