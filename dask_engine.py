# dask_engine.py

import dask.array as da
import imageio.v2 as imageio
import numpy as np
import os
from dask.distributed import Client, LocalCluster
from numba import jit
from PIL import Image
import dask

# This function will be JIT compiled for performance
@jit(nopython=True)
def to_grayscale(image_data):
    """Converts an RGB image to grayscale using the luminosity method."""
    if image_data.ndim == 3 and image_data.shape[2] == 3:
        # Luminosity method: 0.299*R + 0.587*G + 0.114*B
        return (image_data[:, :, 0] * 0.299 +
                image_data[:, :, 1] * 0.587 +
                image_data[:, :, 2] * 0.114).astype(np.uint8)
    elif image_data.ndim == 2:
        return image_data
    else:
        # For formats like RGBA, we just take the first 3 channels
        return (image_data[:, :, 0] * 0.299 +
                image_data[:, :, 1] * 0.587 +
                image_data[:, :, 2] * 0.114).astype(np.uint8)

def _load_and_process_image(image_path):
    """Loads an image, converts it to grayscale, and returns it as a numpy array."""
    img = imageio.imread(image_path)
    if img.ndim == 3:
        if img.shape[2] == 4: # RGBA
            # Convert to RGB by discarding alpha channel
            img = img[:,:,:3]
        img = to_grayscale(img)
    return img

class DaskEngine:
    def __init__(self, thread_count):
        self.thread_count = thread_count
        self.cluster = None
        self.client = None
        self.dask_array = None

    def start_dask_client(self):
        """Starts a local Dask client."""
        if self.client is None:
            self.cluster = LocalCluster(n_workers=self.thread_count, threads_per_worker=1)
            self.client = Client(self.cluster)

    def stop_dask_client(self):
        """Stops the Dask client and cluster."""
        if self.client:
            self.client.close()
            self.cluster.close()
            self.client = None
            self.cluster = None

    def load_images_to_dask_array(self, input_folder):
        """
        Loads PNG images from a folder into a Dask array.
        """
        image_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith('.png')])
        if not image_files:
            raise ValueError("No PNG files found in the input folder.")

        # Read the first image to get dimensions and dtype
        sample_image = _load_and_process_image(image_files[0])
        shape = sample_image.shape
        dtype = sample_image.dtype

        # Create a list of delayed objects
        lazy_images = [dask.delayed(_load_and_process_image)(f) for f in image_files]

        # Create a dask array from the delayed objects
        self.dask_array = da.stack(
            [da.from_delayed(lazy_image, shape=shape, dtype=dtype) for lazy_image in lazy_images],
            axis=0
        )

        # Rechunk the array to the desired chunk size
        self.dask_array = self.dask_array.rechunk((64, 64, 64))


        return self.dask_array

    def get_xy_plane(self, z_index):
        """
        Extracts an XY plane from the Dask array.
        """
        if self.dask_array is None:
            raise ValueError("Dask array not loaded.")
        return self.dask_array[z_index, :, :]

    def get_xz_plane(self, y_index):
        """
        Extracts an XZ plane from the Dask array.
        """
        if self.dask_array is None:
            raise ValueError("Dask array not loaded.")
        return self.dask_array[:, y_index, :]

    def get_yz_plane(self, x_index):
        """
        Extracts a YZ plane from the Dask array.
        """
        if self.dask_array is None:
            raise ValueError("Dask array not loaded.")
        return self.dask_array[:, :, x_index]
