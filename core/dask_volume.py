# -*- coding: utf-8 -*-
"""
Module: dask_volume.py
Author: Jules
Description: Manages the creation of a 3D Dask array from a sequence of 2D slices.
"""

import cv2
import numpy as np
import dask.array as da
from dask import delayed
from pathlib import Path
from typing import Tuple

from core.slice_loader import SliceLoader

class DaskVolume:
    """
    Represents a 3D volume as a Dask array, loaded lazily from a
    directory of 2D image slices.
    """
    def __init__(self, slice_loader: SliceLoader):
        """
        Initializes the DaskVolume and constructs the Dask array.

        Args:
            slice_loader (SliceLoader): An initialized SliceLoader instance.
        """
        if not isinstance(slice_loader, SliceLoader) or len(slice_loader) == 0:
            raise ValueError("A valid SliceLoader instance with found slices is required.")

        self.slice_loader = slice_loader
        self.slice_paths = self.slice_loader.get_slice_list()

        # --- Determine properties from the first slice ---
        first_slice_path = self.slice_paths[0]
        try:
            img = cv2.imread(str(first_slice_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Failed to read or decode the first slice: {first_slice_path}")

            self.height, self.width = img.shape
            self.dtype = img.dtype
            self.num_slices = len(self.slice_paths)
            self.shape = (self.num_slices, self.height, self.width)

        except Exception as e:
            raise RuntimeError(f"Could not process the first slice to determine properties. Error: {e}")

        # --- Build the Dask Array ---
        self.volume = self._create_dask_array()

        print(f"DaskVolume initialized: Shape={self.shape}, "
              f"DType={self.dtype}, Chunks={self.volume.chunksize}")

    def _create_dask_array(self) -> da.Array:
        """
        Creates a 3D Dask array from the image slices.
        """
        # Create a lazy loader for a single image slice
        @delayed
        def load_slice(path):
            return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        # Create a list of delayed objects, one for each slice
        lazy_slices = [load_slice(self.slice_loader.directory / p.name) for p in self.slice_paths]

        # Create a Dask array from the stack of delayed slices
        # We need to tell Dask the shape and dtype of each delayed object
        sample_slice = delayed(cv2.imread(str(self.slice_paths[0]), cv2.IMREAD_GRAYSCALE))

        arrays = [da.from_delayed(s, shape=(self.height, self.width), dtype=self.dtype) for s in lazy_slices]

        # Stack the 2D arrays into a 3D array
        dask_array = da.stack(arrays, axis=0)

        # Rechunk the array according to the user's specification
        return dask_array.rechunk((64, 64, 64))

    def get_xy_slice(self, z_index: int) -> da.Array:
        """Returns a 2D Dask array representing a single XY slice."""
        return self.volume[z_index, :, :]

    def get_xz_slice(self, y_index: int) -> da.Array:
        """Returns a 2D Dask array representing a single XZ slice."""
        return self.volume[:, y_index, :]

    def get_yz_slice(self, x_index: int) -> da.Array:
        """Returns a 2D Dask array representing a single YZ slice."""
        return self.volume[:, :, x_index]
