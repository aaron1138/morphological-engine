import dask.array as da
import dask
import imageio
import numpy as np
import os
from numba import jit

class DaskEngine:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.image_stack = self._load_image_stack()

    def _load_image_stack(self):
        """
        Loads a stack of images from a folder into a Dask array.
        """
        image_files = sorted([os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.lower().endswith(('.png', '.bmp', '.tif', '.tiff'))])
        if not image_files:
            raise ValueError("No image files found in the input folder.")

        # Use dask.delayed to lazy load each image
        lazy_imread = dask.delayed(imageio.imread)
        lazy_images = [lazy_imread(f) for f in image_files]

        # Get the shape and dtype from the first image
        sample_image = imageio.imread(image_files[0])
        shape = (len(lazy_images),) + sample_image.shape
        dtype = sample_image.dtype

        # Create the Dask array
        stack = [da.from_delayed(lazy_image, shape=sample_image.shape, dtype=dtype) for lazy_image in lazy_images]

        return da.stack(stack, axis=0).rechunk((64, 64, 64))

    def get_xy_plane(self, index):
        """
        Extracts an XY plane (a single image) from the stack.
        """
        return self.image_stack[index, :, :]

    def get_xz_plane(self, index):
        """
        Extracts an XZ plane from the stack.
        """
        return self.image_stack[:, :, index]

    def get_yz_plane(self, index):
        """
        Extracts a YZ plane from the stack.
        """
        return self.image_stack[:, index, :]

    @staticmethod
    @jit(nopython=True)
    def _scale_to_uint8(plane):
        """
        Scales a plane to a uint8 array.
        This is a separate function to allow for Numba acceleration.
        """
        if plane.dtype == np.uint8:
            return plane

        plane = plane.astype(np.float32)
        min_val = plane.min()
        max_val = plane.max()
        if max_val == min_val:
            return np.zeros(plane.shape, dtype=np.uint8)

        scaled_plane = 255 * (plane - min_val) / (max_val - min_val)
        return scaled_plane.astype(np.uint8)

    def save_plane(self, plane, filepath):
        """
        Saves a Dask array (a plane) to a file.
        """
        # Ensure the output directory exists
        output_dir = os.path.dirname(filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Compute the plane and scale it to a uint8 array for saving as an image
        computed_plane = plane.compute()
        scaled_plane = self._scale_to_uint8(computed_plane)

        imageio.imwrite(filepath, scaled_plane)
