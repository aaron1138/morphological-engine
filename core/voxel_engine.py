# -*- coding: utf-8 -*-
"""
Module: voxel_engine.py
Author: Gemini
Description: Manages the creation of 3D voxel windows from 2D slices.
             It operates on a sliding window principle to conserve memory and
             provides RAM usage estimates for the processing pipeline.
             This version is updated to use OpenVDB for voxel representation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, List, TYPE_CHECKING

try:
    import pyopenvdb as vdb
except ImportError:
    print("Warning: pyopenvdb is not installed. This module requires it to function.")
    vdb = None

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


class VoxelEngine:
    """
    Creates 3D OpenVDB grids from a list of 2D slice files using a
    sliding window approach.
    """
    def __init__(self, slice_loader: SliceLoader, window_size: int):
        """
        Initializes the VoxelEngine.

        Args:
            slice_loader (SliceLoader): An initialized SliceLoader instance that has
                                        already found and sorted the slice files.
            window_size (int): The number of slices to include in each 3D window (depth).
        """
        if vdb is None:
            raise ImportError("pyopenvdb is not installed. Please follow SETUP.md.")

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
            # Read image as grayscale
            img = cv2.imread(str(first_slice_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Failed to read or decode the first slice: {first_slice_path}")

            self.height, self.width = img.shape
            self.dtype = img.dtype # Typically uint8

        except Exception as e:
            raise RuntimeError(f"Could not process the first slice to determine properties. Error: {e}")

        print(f"VoxelEngine initialized: {self.width}x{self.height}px slices, "
              f"dtype={self.dtype}, window_size={self.window_size}")

    def _numpy_to_openvdb(self, np_array: np.ndarray) -> 'vdb.FloatGrid':
        """
        Converts a dense 3D NumPy array into a sparse OpenVDB FloatGrid.
        The input array values (uint8) are normalized to floats [0.0, 1.0].

        Args:
            np_array (np.ndarray): The input NumPy array, expected to be 3D.

        Returns:
            vdb.FloatGrid: An OpenVDB grid containing the data.
        """
        # Normalize uint8 array (0-255) to float (0.0-1.0)
        if np_array.dtype == np.uint8:
            np_array = np_array.astype(np.float32) / 255.0

        # OpenVDB expects Z, Y, X order, but NumPy arrays are typically indexed
        # as (depth, height, width) which corresponds to (z, y, x).
        # pyopenvdb's from_dense handles this mapping correctly.
        grid = vdb.FloatGrid.from_dense(np_array)
        grid.name = 'density'
        grid.grid_class = vdb.GridClass.FOG_VOLUME
        return grid

    def estimate_ram_usage(self, intermediate_array_factor: int = 4) -> Tuple[float, str]:
        """
        Estimates the RAM required to process a single voxel window.

        NOTE: This estimation is based on a DENSE numpy array. The actual RAM usage
        of the sparse OpenVDB grid will likely be lower, depending on the number
        of active voxels (non-zero values). This serves as a worst-case estimate.

        Args:
            intermediate_array_factor (int): A multiplier to account for temporary
                arrays created during processing.

        Returns:
            Tuple[float, str]: A tuple containing the estimated RAM amount and its unit.
        """
        bytes_per_pixel = np.dtype(self.dtype).itemsize

        # Memory for one window (in bytes)
        base_window_bytes = self.width * self.height * self.window_size * bytes_per_pixel

        # Estimated total memory including intermediate arrays
        estimated_bytes = base_window_bytes * intermediate_array_factor

        # Convert to a human-readable format
        if estimated_bytes < 1024**3: # Less than 1 GB
            ram_mb = estimated_bytes / (1024**2)
            return round(ram_mb, 2), "MB"
        else:
            ram_gb = estimated_bytes / (1024**3)
            return round(ram_gb, 2), "GB"

    def iter_windows(self) -> Generator['vdb.FloatGrid', None, None]:
        """
        A generator that yields successive 3D voxel windows as OpenVDB FloatGrids.

        This is the core memory-saving feature of the engine.

        Yields:
            vdb.FloatGrid: An OpenVDB grid representing the voxel window.
        """
        slice_paths = self.slice_loader.get_slice_list()
        num_windows = self.total_slices - self.window_size + 1

        for i in range(num_windows):
            window_np = np.zeros((self.window_size, self.height, self.width), dtype=self.dtype)
            window_slice_paths = slice_paths[i : i + self.window_size]

            for z, slice_path in enumerate(window_slice_paths):
                try:
                    img = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Could not read slice {slice_path}. Skipping.")
                        continue
                    if img.shape != (self.height, self.width):
                        print(f"Warning: Slice {slice_path} has mismatched dimensions. Skipping.")
                        continue
                    window_np[z, :, :] = img
                except Exception as e:
                    print(f"Error loading slice {slice_path}: {e}. Skipping.")

            # Convert the NumPy window to an OpenVDB grid and yield it
            voxel_grid = self._numpy_to_openvdb(window_np)
            yield voxel_grid

# --- Example Usage ---
if __name__ == '__main__':
    from core.slice_loader import SliceLoader
    import time

    print("--- VoxelEngine with OpenVDB Test ---")

    if vdb is None:
        print("\nCannot run test: pyopenvdb is not installed.")
    else:
        # --- Setup a test environment ---
        test_dir = Path("./temp_voxel_engine_test_dir")
        test_dir.mkdir(exist_ok=True)

        # Create 20 dummy slice files for testing
        num_dummy_files = 20
        print(f"\nCreating {num_dummy_files} dummy slice files...")
        dummy_shape = (100, 150) # Small dimensions for testing
        for i in range(1, num_dummy_files + 1):
            fname = test_dir / f"slice_{i:04d}.png"
            img = np.zeros(dummy_shape, dtype=np.uint8)
            # Create a non-empty region to have some active voxels
            if i > 5 and i < 15:
                cv2.rectangle(img, (20, 20), (130, 80), (200), -1)
            cv2.putText(img, str(i), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255), 3)
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
                print(f"  -> Yielded window {i+1}: "
                      f"Grid Type={type(voxel_grid)}, "
                      f"Active Voxels={voxel_grid.active_voxel_count()}, "
                      f"Name='{voxel_grid.name}'")
                window_count += 1
                assert isinstance(voxel_grid, vdb.FloatGrid)
                assert voxel_grid.is_class(vdb.GridClass.FOG_VOLUME)

            end_time = time.time()

            print(f"\nSuccessfully iterated through {window_count} OpenVDB grids in {end_time - start_time:.2f} seconds.")

        except (ValueError, RuntimeError, FileNotFoundError, ImportError) as e:
            print(f"\nAn error occurred: {e}")
            print("Please ensure you have compiled and installed pyopenvdb correctly.")
        finally:
            # --- Clean up ---
            print("\nCleaning up test directory...")
            for f in test_dir.glob('*.png'):
                f.unlink()
            if test_dir.exists():
                try:
                    test_dir.rmdir()
                except OSError as e:
                    print(f"Error removing test directory: {e}")
            print("Cleanup complete.")
