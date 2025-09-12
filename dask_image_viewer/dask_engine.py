import os
import dask.array as da
import imageio.v2 as imageio
from dask import delayed
import numpy as np
import re

class DaskEngine:
    def __init__(self, input_folder, thread_count):
        self.input_folder = input_folder
        self.dask_array = None
        self.thread_count = thread_count

    def load_images(self):
        """
        Loads images from the input folder into a Dask array.
        """
        # Get a sorted list of image files
        image_files = self._get_sorted_image_files()
        if not image_files:
            raise ValueError("No image files found in the specified folder.")

        # Lazily read each image and stack them into a Dask array
        sample_image = imageio.imread(image_files[0])

        lazy_images = [delayed(imageio.imread)(f) for f in image_files]

        # Check if the images are grayscale or RGB and handle accordingly
        if sample_image.ndim == 2: # Grayscale
            shape = (len(lazy_images),) + sample_image.shape
            dtype = sample_image.dtype
        elif sample_image.ndim == 3: # RGB or other multi-channel
            # We'll take the first channel, assuming grayscale data is encoded in RGB
            lazy_images = [delayed(lambda x: imageio.imread(x)[:,:,0])(f) for f in image_files]
            shape = (len(lazy_images),) + sample_image.shape[:2]
            dtype = sample_image.dtype
        else:
            raise ValueError("Unsupported image format.")

        self.dask_array = da.stack(lazy_images, axis=0).rechunk({0: 64, 1: 64, 2: 64})

        return self.dask_array

    def _get_sorted_image_files(self):
        """
        Gets a numerically sorted list of PNG files from the input folder.
        """
        files = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.lower().endswith('.png')]

        def get_numeric_part(filename):
            match = re.search(r'(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else float('inf')

        return sorted(files, key=get_numeric_part)

    def get_xy_plane(self, z_index):
        """
        Extracts an XY plane (a single image) from the Dask array.
        """
        if self.dask_array is None:
            return None
        return self.dask_array[z_index, :, :]

    def get_xz_plane(self, y_index):
        """
        Extracts an XZ plane from the Dask array.
        """
        if self.dask_array is None:
            return None
        return self.dask_array[:, y_index, :]

    def get_yz_plane(self, x_index):
        """
        Extracts a YZ plane from the Dask array.
        """
        if self.dask_array is None:
            return None
        return self.dask_array[:, :, x_index]
