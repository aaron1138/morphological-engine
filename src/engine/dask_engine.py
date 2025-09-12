# src/engine/dask_engine.py

import os
import glob
import re
import dask
import dask.array as da
import numpy as np
import imageio.v2 as imageio
from numba import jit

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """Key for natural sorting of strings."""
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

@jit(nopython=True)
def to_grayscale(rgb_image):
    """Converts an RGB image to grayscale using the luminosity method."""
    # Using standard weights for RGB to grayscale conversion
    gray = 0.2126 * rgb_image[:, :, 0] + 0.7152 * rgb_image[:, :, 1] + 0.0722 * rgb_image[:, :, 2]
    return gray.astype(np.uint8)

class DaskEngine:
    """
    Handles the core Dask operations for loading and accessing the 3D image stack.
    """
    def __init__(self):
        self.stack = None

    def load_image_stack(self, input_folder: str, file_pattern: str = "*.png", chunks: tuple = (64, 64, 64)):
        """
        Loads a folder of images as a 3D Dask array.

        Args:
            input_folder (str): The path to the folder containing the images.
            file_pattern (str): The pattern to match image files (e.g., "*.png").
            chunks (tuple): The desired chunk size for the Dask array (Z, Y, X).
        """
        filepaths = sorted(glob.glob(os.path.join(input_folder, file_pattern)), key=natural_sort_key)
        if not filepaths:
            raise ValueError(f"No images found in '{input_folder}' matching '{file_pattern}'")

        # Lazily read the first image to determine shape and dtype
        sample_image = imageio.imread(filepaths[0])

        # Function to read and process a single image
        def read_and_process_image(fp):
            img = imageio.imread(fp)
            if len(img.shape) == 3: # Check if it's an RGB or RGBA image
                if img.shape[2] == 3: # RGB
                    return to_grayscale(img)
                elif img.shape[2] == 4: # RGBA, convert to RGB first
                    # Numba jit compiled functions don't support slicing in this manner,
                    # so we slice the array before passing it to the function.
                    return to_grayscale(img[:, :, :3])
            # Assume it's already grayscale if not 3 channels
            return img

        # Get the shape from a processed sample
        processed_sample = read_and_process_image(filepaths[0])
        final_shape = processed_sample.shape
        dtype = processed_sample.dtype

        # Create a list of delayed Dask objects
        lazy_images = [dask.delayed(read_and_process_image)(fp) for fp in filepaths]

        # Create Dask arrays from the delayed objects
        dask_arrays = [da.from_delayed(lazy_image, shape=final_shape, dtype=dtype) for lazy_image in lazy_images]

        # Stack the arrays along the Z-axis (the first dimension)
        self.stack = da.stack(dask_arrays, axis=0)

        # Re-chunk the stack to the desired 3D chunking scheme
        # The chunks are specified in (Z, Y, X) order
        self.stack = self.stack.rechunk(chunks)

        print(f"Dask array created with shape: {self.stack.shape}")
        print(f"Chunks: {self.stack.chunksize}")

        return self.stack

    def get_xy_plane(self, z_index: int):
        """
        Extracts a 2D XY plane (a single original image) from the stack.

        Args:
            z_index (int): The index of the layer to extract.

        Returns:
            A Dask array representing the XY plane.
        """
        if self.stack is None:
            raise RuntimeError("Image stack has not been loaded.")
        return self.stack[z_index, :, :]

    def get_xz_plane(self, y_index: int):
        """
        Extracts a 2D XZ orthogonal plane from the stack.

        Args:
            y_index (int): The Y-coordinate for the slice.

        Returns:
            A Dask array representing the XZ plane.
        """
        if self.stack is None:
            raise RuntimeError("Image stack has not been loaded.")
        return self.stack[:, y_index, :]

    def get_yz_plane(self, x_index: int):
        """
        Extracts a 2D YZ orthogonal plane from the stack.

        Args:
            x_index (int): The X-coordinate for the slice.

        Returns:
            A Dask array representing the YZ plane.
        """
        if self.stack is None:
            raise RuntimeError("Image stack has not been loaded.")
        return self.stack[:, :, x_index]
