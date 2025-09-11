# -*- coding: utf-8 -*-
"""
Module: voxel_engine.py
Author: Gemini
Description: Manages the creation of a 3D Dask array from a directory of 2D slices.
"""

import cv2
import numpy as np
import dask.array as da
from pathlib import Path
from typing import List, Tuple
from dask import delayed

# Assuming slice_loader.py is in the same directory or a reachable path.
try:
    # CORRECTED IMPORT: Use absolute import from the project root.
    from core.slice_loader import SliceLoader
except ImportError:
    # This allows the module to be run standalone for testing,
    # assuming a mock or the actual class is available.
    print("Warning: Could not import SliceLoader. Standalone testing may be affected.")
    # Define a dummy class if it's not found, for type hinting purposes.
    class SliceLoader:
        def get_slice_list(self) -> List[Path]: return []
        def __len__(self) -> int: return 0

# We need read_image from the data_io module
from data_io.image import read_image


class VoxelEngine:
    """
    Manages the creation of a 3D Dask array from a directory of 2D slices.
    """
    def __init__(self, slice_loader: SliceLoader, chunk_size: Tuple[int, int, int] = (64, 64, 64)):
        """
        Initializes the VoxelEngine.

        Args:
            slice_loader (SliceLoader): An initialized SliceLoader instance.
            chunk_size (Tuple[int, int, int]): The desired chunk size for the Dask array (depth, height, width).
        """
        if not isinstance(slice_loader, SliceLoader) or len(slice_loader) == 0:
            raise ValueError("A valid SliceLoader instance with found slices is required.")

        self.slice_loader = slice_loader
        self.total_slices = len(slice_loader)
        self.chunk_size = chunk_size

        # --- Determine properties from the first slice ---
        first_slice_path = self.slice_loader.get_slice_list()[0]
        try:
            img = read_image(first_slice_path)
            self.height, self.width = img.shape
            self.dtype = img.dtype
        except Exception as e:
            raise RuntimeError(f"Could not process the first slice to determine properties. Error: {e}")

        print(f"VoxelEngine initialized: {self.width}x{self.height}px slices, "
              f"{self.total_slices} total slices, dtype={self.dtype}, chunk_size={self.chunk_size}")

        self._create_dask_array()

    def _create_dask_array(self):
        """
        Creates the Dask array from the image slices.
        """
        # Create a list of delayed read_image calls
        lazy_reads = [delayed(read_image)(path) for path in self.slice_loader.get_slice_list()]

        # Create a dask array from each delayed read
        dask_arrays = [da.from_delayed(lazy_read, shape=(self.height, self.width), dtype=self.dtype)
                       for lazy_read in lazy_reads]

        # Stack the 2D arrays into a 3D array
        self.dask_array = da.stack(dask_arrays, axis=0)

        # Rechunk the array to the desired chunk size
        self.dask_array = self.dask_array.rechunk(self.chunk_size)

        print(f"Dask array created with shape: {self.dask_array.shape} and "
              f"chunks: {self.dask_array.chunksize}")

    def get_dask_array(self) -> da.Array:
        """Returns the full 3D Dask array representing the voxel volume."""
        return self.dask_array

    def get_orthogonal_view(self, axis: int, slice_index: int) -> np.ndarray:
        """
        Computes and returns a 2D orthogonal slice from the Dask array.

        Args:
            axis (int): The axis to slice along (0=XY, 1=XZ, 2=YZ).
            slice_index (int): The index of the slice to retrieve.

        Returns:
            np.ndarray: The computed 2D slice.
        """
        if axis == 0: # XY plane
            view = self.dask_array[slice_index, :, :]
        elif axis == 1: # XZ plane
            view = self.dask_array[:, slice_index, :]
        elif axis == 2: # YZ plane
            view = self.dask_array[:, :, slice_index]
        else:
            raise ValueError("Axis must be 0 (XY), 1 (XZ), or 2 (YZ)")

        print(f"Computing orthogonal view: axis={axis}, index={slice_index}")
        return view.compute()

if __name__ == '__main__':
    import sys
    from pathlib import Path
    # Add project root to path to allow imports from other modules
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from data_io.image import write_image
    import time

    print("--- VoxelEngine Dask Test ---")

    test_dir = Path("./temp_dask_engine_test")
    if test_dir.exists():
        for f in test_dir.iterdir():
            f.unlink()
        test_dir.rmdir()
    test_dir.mkdir(exist_ok=True)

    # Create dummy slice files for testing
    num_files = 100
    shape = (128, 128)
    print(f"\nCreating {num_files} dummy slice files...")
    for i in range(num_files):
        fname = test_dir / f"slice_{i:04d}.png"
        img = np.full(shape, fill_value=i, dtype=np.uint8)
        write_image(fname, img)

    print("Dummy files created.")

    try:
        # We have to use the real SliceLoader for the test now
        loader = SliceLoader(str(test_dir))
        engine = VoxelEngine(loader, chunk_size=(64, 64, 64))

        dask_vol = engine.get_dask_array()
        print(f"\nRetrieved Dask array: {dask_vol}")

        # --- Test orthogonal view extraction ---
        print("\n--- Testing Orthogonal Views ---")

        # Get an XZ slice (axis 1)
        xz_slice = engine.get_orthogonal_view(axis=1, slice_index=10)
        print(f"  - Retrieved XZ slice of shape: {xz_slice.shape}")
        assert xz_slice.shape == (num_files, shape[1])

        # Get a YZ slice (axis 2)
        yz_slice = engine.get_orthogonal_view(axis=2, slice_index=20)
        print(f"  - Retrieved YZ slice of shape: {yz_slice.shape}")
        assert yz_slice.shape == (num_files, shape[0])

        print("\nOrthogonal view test PASSED.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # --- Clean up ---
        print("\nCleaning up test directory...")
        if test_dir.exists():
            for f in test_dir.iterdir():
                f.unlink()
            test_dir.rmdir()
        print("Cleanup complete.")
