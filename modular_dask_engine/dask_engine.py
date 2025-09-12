import os
import dask.array as da
import imageio.v2 as imageio
import numpy as np
import re
from dask.distributed import Client, LocalCluster
from numba import jit

@jit(nopython=True)
def numba_accelerated_function(chunk):
    """
    A placeholder for a Numba-accelerated function that performs a pixel-wise operation.
    """
    # Example operation: invert the image chunk
    return 255 - chunk

class DaskEngine:
    def __init__(self, config):
        self.config = config
        self.cluster = None
        self.client = None
        self.dask_array = None

    def start_dask_client(self):
        """Starts a local Dask cluster and client."""
        if self.client and self.client.status == 'running':
            print("Dask client is already running.")
            return

        self.cluster = LocalCluster(n_workers=self.config.thread_count, threads_per_worker=1)
        self.client = Client(self.cluster)
        print(f"Dask client started with {self.config.thread_count} workers.")
        print(f"Dask dashboard link: {self.client.dashboard_link}")

    def stop_dask_client(self):
        """Stops the Dask client and cluster."""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()
        self.client = None
        self.cluster = None
        print("Dask client stopped.")

    def load_images_to_dask_array(self, image_folder):
        """
        Loads a folder of PNG images into a Dask array.
        """
        numeric_pattern = re.compile(r'(\d+)\.\w+$')
        def get_numeric_part(filename):
            match = numeric_pattern.search(filename)
            return int(match.group(1)) if match else float('inf')

        image_files = sorted(
            [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith('.png')],
            key=get_numeric_part
        )

        if not image_files:
            raise ValueError("No PNG images found in the specified folder.")

        # Lazily read each image and stack them into a Dask array
        lazy_images = [da.from_delayed(da.utils.delayed(imageio.imread)(f), shape=imageio.imread(image_files[0]).shape, dtype=np.uint8) for f in image_files]

        # Determine chunk sizes
        chunk_sizes = (self.config.chunk_size_z, self.config.chunk_size_y, self.config.chunk_size_x)

        self.dask_array = da.stack(lazy_images, axis=0).rechunk(chunk_sizes)

        print(f"Dask array created with shape: {self.dask_array.shape} and chunks: {self.dask_array.chunksize}")
        return self.dask_array

    def get_xy_plane(self, z_index):
        """
        Extracts an XY plane (a single image) from the Dask array.
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

    def apply_numba_function(self):
        """
        Applies the Numba-accelerated function to the Dask array.
        """
        if self.dask_array is None:
            raise ValueError("Dask array not loaded.")

        # The map_blocks function applies a function to each block of a Dask array.
        # This is where you would apply your custom Numba-accelerated functions.
        processed_array = self.dask_array.map_blocks(numba_accelerated_function, dtype=np.uint8)

        # In a real application, you would then save this processed array to disk.
        # For this example, we'll just compute it and return it.
        return processed_array.compute()
