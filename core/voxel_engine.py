# -*- coding: utf-8 -*-
"""
Module: voxel_engine.py
Author: Jules (Refactored for OpenVDB)
Description: Manages the creation of 3D voxel windows from 2D slices.
             This version uses OpenVDB to represent the voxel data, allowing for
             sparse data structures and advanced 3D operations.
"""

import cv2
import numpy as np
import openvdb
from pathlib import Path
from typing import Generator, Tuple, List

from core.slice_loader import SliceLoader

# Initialize the OpenVDB library
openvdb.initialize()

class VoxelEngine:
    """
    Creates 3D OpenVDB grids from a list of 2D slice files using a
    sliding window approach.
    """
    def __init__(self, slice_loader: SliceLoader, window_size: int):
        """
        Initializes the VoxelEngine.

        Args:
            slice_loader (SliceLoader): An initialized SliceLoader instance.
            window_size (int): The number of slices for the 3D window depth.
        """
        if not isinstance(slice_loader, SliceLoader) or len(slice_loader) == 0:
            raise ValueError("A valid SliceLoader instance with found slices is required.")

        if not (3 <= window_size <= len(slice_loader)):
             raise ValueError(f"Window size must be at least 3 and no larger than the "
                              f"number of slices ({len(slice_loader)}).")

        self.slice_loader = slice_loader
        self.window_size = window_size
        self.total_slices = len(slice_loader)

        # --- Determine properties from the first slice ---
        first_slice_path = self.slice_loader[0]
        try:
            img = cv2.imread(str(first_slice_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Failed to read or decode the first slice: {first_slice_path}")

            self.height, self.width = img.shape
            self.dtype = img.dtype  # Typically uint8

        except Exception as e:
            raise RuntimeError(f"Could not process the first slice to determine properties. Error: {e}")

        print(f"VoxelEngine initialized: {self.width}x{self.height}px slices, "
              f"dtype={self.dtype}, window_size={self.window_size}, backend=OpenVDB")

    def estimate_ram_usage(self, intermediate_array_factor: int = 4) -> Tuple[float, str]:
        """
        Estimates the RAM for a single DENSE voxel window.

        This provides a worst-case scenario for the GUI to prevent users from
        starting a process that would exhaust system memory. The actual memory
        usage with sparse OpenVDB grids will likely be lower.

        Args:
            intermediate_array_factor (int): A multiplier for temporary arrays.

        Returns:
            Tuple[float, str]: The estimated RAM amount and its unit (MB or GB).
        """
        bytes_per_pixel = np.dtype(np.float32).itemsize # We convert to float for VDB

        # Memory for one dense window (in bytes)
        dense_window_bytes = self.width * self.height * self.window_size * bytes_per_pixel

        estimated_bytes = dense_window_bytes * intermediate_array_factor

        if estimated_bytes < 1024**3:
            ram_mb = estimated_bytes / (1024**2)
            return round(ram_mb, 2), "MB"
        else:
            ram_gb = estimated_bytes / (1024**3)
            return round(ram_gb, 2), "GB"

    def iter_windows(self) -> Generator[openvdb.FloatGrid, None, None]:
        """
        A generator that yields successive 3D voxel windows as OpenVDB FloatGrids.

        Yields:
            openvdb.FloatGrid: An OpenVDB grid containing the voxel data for the window.
                               The data is normalized to floats [0.0, 1.0].
        """
        slice_paths = self.slice_loader.get_slice_list()
        num_windows = self.total_slices - self.window_size + 1

        for i in range(num_windows):
            # Pre-allocate a NumPy array to temporarily hold the window's slice data
            window_np = np.zeros((self.window_size, self.height, self.width), dtype=np.float32)

            window_slice_paths = slice_paths[i : i + self.window_size]

            for z, slice_path in enumerate(window_slice_paths):
                try:
                    img = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
                    if img is None or img.shape != (self.height, self.width):
                        print(f"Warning: Skipping invalid slice {slice_path}")
                        continue

                    # Normalize uint8 [0, 255] to float [0.0, 1.0]
                    window_np[z, :, :] = img.astype(np.float32) / 255.0

                except Exception as e:
                    print(f"Error loading slice {slice_path}: {e}. Skipping.")

            # Create a new FloatGrid and copy the NumPy array data into it.
            # OpenVDB uses a different coordinate system convention (Z, Y, X),
            # so we copy the array directly as its layout matches this.
            grid = openvdb.FloatGrid()
            grid.copyFromArray(window_np)
            grid.name = "density"
            grid.gridClass = openvdb.GridClass.FOG_VOLUME

            # Set up the grid's transform. Assume 1 voxel = 1 unit.
            grid.transform = openvdb.createLinearTransform(voxelSize=1.0)

            yield grid

# --- Example Usage ---
if __name__ == '__main__':
    from core.slice_loader import SliceLoader
    import time

    print("--- VoxelEngine (OpenVDB Backend) Test ---")

    test_dir = Path("./temp_voxel_engine_test_dir")
    if not test_dir.exists():
        test_dir.mkdir(exist_ok=True)

    num_dummy_files = 20
    print(f"\nCreating {num_dummy_files} dummy slice files...")
    dummy_shape = (100, 150)
    for i in range(1, num_dummy_files + 1):
        fname = test_dir / f"slice_{i:04d}.png"
        img = np.zeros(dummy_shape, dtype=np.uint8)
        # Create a pattern that won't be entirely sparse
        cv2.rectangle(img, (20, 20), (80, 80), (128), -1)
        cv2.putText(img, str(i), (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
        cv2.imwrite(str(fname), img)

    print("Dummy files created.")

    try:
        loader = SliceLoader(str(test_dir))
        WINDOW_DEPTH = 5
        engine = VoxelEngine(loader, window_size=WINDOW_DEPTH)

        ram, unit = engine.estimate_ram_usage()
        print(f"\nEstimated RAM per window (dense): {ram} {unit}")

        print(f"\nIterating through voxel windows of size {WINDOW_DEPTH}...")
        start_time = time.time()

        window_count = 0
        for i, voxel_grid in enumerate(engine.iter_windows()):
            mem_kb = voxel_grid.memUsage() / 1024
            print(f"  -> Yielded window {i+1}: "
                  f"Grid Type={type(voxel_grid).__name__}, "
                  f"Active Voxels={voxel_grid.activeVoxelCount()}, "
                  f"Mem Usage={mem_kb:.2f} KB")
            window_count += 1

        end_time = time.time()
        print(f"\nSuccessfully iterated through {window_count} windows in {end_time - start_time:.2f} seconds.")

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up test directory...")
        for f in test_dir.iterdir():
            f.unlink()
        test_dir.rmdir()
        print("Cleanup complete.")
