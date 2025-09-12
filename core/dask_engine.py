# core/dask_engine.py
import dask
import dask.array as da
from numba import jit
import numpy as np
import cv2
import os

class DaskEngine:
    def __init__(self, chunk_size=(64, 64, 64)):
        self.chunk_size = chunk_size

    def load_images_to_dask_array(self, image_paths: list[str]):
        """
        Lazily loads a list of images and stacks them into a Dask array.

        Args:
            image_paths: A list of paths to the image files.

        Returns:
            A Dask array representing the stack of images.
        """
        if not image_paths:
            raise ValueError("The list of image paths cannot be empty.")

        # Use dask.delayed to read each image lazily
        @dask.delayed
        def read_image(path):
            if not os.path.exists(path):
                # Return a sensible default or raise an error.
                # For now, let's return None and filter it out later.
                return None

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if img is None:
                return None # Could not read image

            # Handle different image formats (e.g., RGB, RGBA) by converting to grayscale
            if len(img.shape) == 3 and img.shape[2] in [3, 4]:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Ensure the image is 8-bit
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            return img

        # Create a list of delayed image objects
        delayed_images = [read_image(path) for path in image_paths]

        # Get the shape and dtype from the first image to build the Dask array
        sample_img_path = image_paths[0]
        try:
            sample_img = cv2.imread(sample_img_path, cv2.IMREAD_UNCHANGED)
            if sample_img is None:
                raise ValueError(f"Could not read the first image to determine shape: {sample_img_path}")
            if len(sample_img.shape) == 3:
                h, w, _ = sample_img.shape
            else:
                h, w = sample_img.shape
            dtype = np.uint8 # We are standardizing to 8-bit grayscale
        except Exception as e:
            raise RuntimeError(f"Failed to read sample image to determine stack properties: {e}")

        # Create a Dask array from the list of delayed objects
        stack = [da.from_delayed(img, shape=(h, w), dtype=dtype) for img in delayed_images]
        dask_array = da.stack(stack, axis=0)

        # Rechunk the array to the desired 3D chunking
        return dask_array.rechunk(self.chunk_size)

    def get_xy_plane(self, dask_array, z_index):
        """Extracts an XY plane (a single original image)."""
        return dask_array[z_index, :, :]

    def get_xz_plane(self, dask_array, y_index):
        """Extracts an XZ plane."""
        return dask_array[:, y_index, :]

    def get_yz_plane(self, dask_array, x_index):
        """Extracts a YZ plane."""
        return dask_array[:, :, x_index]

@jit(nopython=True)
def numba_accelerated_function(data):
    """
    A placeholder for a Numba-accelerated function.
    This function must be called with a NumPy array, not a Dask array.
    You would typically use this inside a dask.array.map_blocks call.
    """
    # Example operation
    return (data * 2).astype(np.uint8)
