import os
import re
import dask.array as da
from dask import delayed
import imageio.v2 as imageio
from skimage.color import rgb2gray
from skimage import io
import numpy as np
from numba import jit

class DaskEngine:
    def __init__(self, config):
        self.config = config
        self.stack = None

    def load_images_to_dask_stack(self, image_folder: str):
        """
        Loads a folder of images into a 3D Dask array with natural sorting.
        """
        def natural_sort_key(s):
            """A key for natural sorting of filenames."""
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        all_files = os.listdir(image_folder)
        image_filenames = [f for f in all_files if f.lower().endswith(('.png', '.tif', '.tiff', '.bmp'))]

        if not image_filenames:
            raise ValueError("No image files found in the specified folder.")

        image_filenames.sort(key=natural_sort_key)
        image_paths = [os.path.join(image_folder, f) for f in image_filenames]

        # Lazily load the first image to get shape and dtype
        sample_image = imageio.imread(image_paths[0])
        if sample_image.ndim == 3: # Convert RGB to grayscale if necessary
            sample_image = (rgb2gray(sample_image) * 255).astype(np.uint8)

        shape = (len(image_paths), sample_image.shape[0], sample_image.shape[1])
        dtype = sample_image.dtype

        @delayed
        def read_image(filepath):
            img = imageio.imread(filepath)
            if img.ndim == 3:
                # Ensure consistent conversion to 8-bit grayscale
                return (rgb2gray(img) * 255).astype(dtype)
            return img

        lazy_images = [read_image(f) for f in image_paths]

        dask_images = [
            da.from_delayed(lazy_image, shape=shape[1:], dtype=dtype)
            for lazy_image in lazy_images
        ]

        self.stack = da.stack(dask_images, axis=0)

        self.stack = self.stack.rechunk({
            0: self.config.chunk_size_z,
            1: self.config.chunk_size_y,
            2: self.config.chunk_size_x
        })

        return self.stack

    def get_xy_plane(self, z_index: int):
        """
        Extracts a 2D (XY) plane from the stack.
        """
        if self.stack is None:
            raise RuntimeError("Image stack has not been loaded.")
        return self.stack[z_index, :, :]

    def get_xz_plane(self, y_index: int):
        """
        Extracts a 2D (XZ) plane from the stack.
        """
        if self.stack is None:
            raise RuntimeError("Image stack has not been loaded.")
        return self.stack[:, y_index, :]

    def get_yz_plane(self, x_index: int):
        """
        Extracts a 2D (YZ) plane from the stack.
        """
        if self.stack is None:
            raise RuntimeError("Image stack has not been loaded.")
        return self.stack[:, :, x_index]

    @staticmethod
    @jit(nopython=True)
    def placeholder_numba_function(data):
        """
        A placeholder for a Numba-accelerated function that could be
        applied to array chunks.
        """
        return 255 - data
