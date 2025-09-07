# -*- coding: utf-8 -*-
"""
Module: voxel_engine.py
Author: Gemini
Description: Manages the creation of 3D voxel windows from 2D slices.
             It operates on a sliding window principle to conserve memory and
             provides RAM usage estimates for the processing pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, List

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

try:
    import pyopenvdb as vdb
except ImportError:
    print("Warning: pyopenvdb is not installed. OpenVDB functionality will be unavailable.")
    vdb = None


class VoxelEngine:
    """
    Creates 3D voxel data blocks from a list of 2D slice files using a
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

    def estimate_ram_usage(self, intermediate_array_factor: int = 4) -> Tuple[float, str]:
        """
        Estimates the RAM required to process a single voxel window.

        This is a crucial function for the GUI to prevent users from starting a
        process that would exhaust system memory.

        Args:
            intermediate_array_factor (int): A multiplier to account for temporary
                arrays created during processing (e.g., for gradients, masks).
                A factor of 4 suggests the peak memory might be 4x the size of
                the input window itself.

        Returns:
            Tuple[float, str]: A tuple containing the estimated RAM amount and its unit (MB or GB).
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

    def iter_windows(self) -> Generator[np.ndarray, None, None]:
        """
        A generator that yields successive 3D voxel windows from the slice files.

        This is the core memory-saving feature of the engine.

        Yields:
            np.ndarray: A 3D NumPy array of shape (window_size, height, width)
                        and the determined dtype.
        """
        slice_paths = self.slice_loader.get_slice_list()

        # We can create (total_slices - window_size + 1) windows
        num_windows = self.total_slices - self.window_size + 1

        for i in range(num_windows):
            # Pre-allocate memory for the current window
            window = np.zeros((self.window_size, self.height, self.width), dtype=self.dtype)

            # Get the file paths for the current window
            window_slice_paths = slice_paths[i : i + self.window_size]

            for z, slice_path in enumerate(window_slice_paths):
                try:
                    # Load the slice image
                    img = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Could not read slice {slice_path}. Skipping.")
                        continue

                    # Basic validation
                    if img.shape != (self.height, self.width):
                        print(f"Warning: Slice {slice_path} has mismatched dimensions. Skipping.")
                        continue

                    window[z, :, :] = img
                except Exception as e:
                    print(f"Error loading slice {slice_path}: {e}. Skipping.")

            # Yield the complete 3D window for processing
            yield window

    # --- OpenVDB Methods ---

    @staticmethod
    def save_grid(grid: 'vdb.Grid', filepath: str):
        """Saves a single OpenVDB grid to a .vdb file."""
        if vdb is None:
            raise ImportError("pyopenvdb is not installed.")
        vdb.write(filepath, grids=[grid])
        print(f"Grid '{grid.name}' saved to {filepath}")

    @staticmethod
    def load_grids(filepath: str) -> List['vdb.Grid']:
        """Loads a list of grids from a .vdb file."""
        if vdb is None:
            raise ImportError("pyopenvdb is not installed.")
        return vdb.read(filepath)

    def numpy_to_vdb_grid(self, np_array: np.ndarray, grid_name: str = "density") -> 'vdb.FloatGrid':
        """
        Converts a NumPy array to an OpenVDB FloatGrid.
        The input array is expected to be in (depth, height, width) order.
        OpenVDB expects (z, y, x) which corresponds to this order.
        """
        if vdb is None:
            raise ImportError("pyopenvdb is not installed.")

        # Ensure the numpy array is in a C-contiguous layout.
        np_array = np.ascontiguousarray(np_array)

        grid = vdb.FloatGrid()
        grid.copyFromArray(np_array.astype(float))
        grid.name = grid_name
        grid.gridClass = vdb.GridClass.FOG_VOLUME # Represents a density volume

        # Set voxel size and transform if needed (optional)
        # grid.transform = vdb.createLinearTransform(voxelSize=1.0)

        return grid

    def iter_vdb_grids(self, grid_name_prefix: str = "density") -> Generator['vdb.FloatGrid', None, None]:
        """
        A generator that yields successive 3D voxel windows as OpenVDB FloatGrids.
        """
        if vdb is None:
            raise ImportError("pyopenvdb is not installed.")

        for i, window in enumerate(self.iter_windows()):
            grid_name = f"{grid_name_prefix}_{i}"
            yield self.numpy_to_vdb_grid(window, grid_name=grid_name)

# --- Example Usage ---
if __name__ == '__main__':
    from core.slice_loader import SliceLoader
    import time

    print("--- VoxelEngine Test ---")

    # --- Setup a test environment ---
    test_dir = Path("./temp_voxel_engine_test_dir")
    if not test_dir.exists():
        test_dir.mkdir(exist_ok=True)

    # Create 20 dummy slice files for testing
    num_dummy_files = 20
    print(f"\nCreating {num_dummy_files} dummy slice files...")
    dummy_shape = (100, 150) # Small dimensions for testing
    for i in range(1, num_dummy_files + 1):
        # Create a blank image with a number written on it
        fname = test_dir / f"slice_{i:04d}.png"
        img = np.zeros(dummy_shape, dtype=np.uint8)
        cv2.putText(img, str(i), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255), 3)
        cv2.imwrite(str(fname), img)

    print("Dummy files created.")

    try:
        # 1. Initialize the SliceLoader
        loader = SliceLoader(str(test_dir))

        # 2. Initialize the VoxelEngine
        WINDOW_DEPTH = 5
        engine = VoxelEngine(loader, window_size=WINDOW_DEPTH)

        # 3. Get RAM estimation
        ram, unit = engine.estimate_ram_usage()
        print(f"\nEstimated RAM per window: {ram} {unit}")

        # 4. Iterate through the voxel windows (NumPy)
        print(f"\nIterating through NumPy voxel windows of size {WINDOW_DEPTH}...")
        start_time = time.time()

        window_count = 0
        for i, voxel_window in enumerate(engine.iter_windows()):
            if i == 0: # Only print for the first one to keep output clean
                print(f"  -> Yielded NumPy window {i+1}: "
                      f"Shape={voxel_window.shape}, DType={voxel_window.dtype}, "
                      f"Mean value={voxel_window.mean():.2f}")
            window_count += 1
        end_time = time.time()
        print(f"Successfully iterated through {window_count} NumPy windows in {end_time - start_time:.2f} seconds.")

        # 5. Test OpenVDB functionality if available
        if vdb:
            print("\n--- OpenVDB Functionality Test ---")

            # Get the first grid from the VDB generator
            first_vdb_grid = next(engine.iter_vdb_grids())
            print(f"Successfully created a VDB grid named '{first_vdb_grid.name}'")
            print(f"  - Grid Class: {first_vdb_grid.gridClass}")
            print(f"  - Active Voxels: {first_vdb_grid.activeVoxelCount()}")

            # Save and load the grid
            vdb_file_path = test_dir / "test_grid.vdb"
            VoxelEngine.save_grid(first_vdb_grid, str(vdb_file_path))

            loaded_grids = VoxelEngine.load_grids(str(vdb_file_path))
            print(f"Loaded {len(loaded_grids)} grid(s) from {vdb_file_path}")

            loaded_grid = loaded_grids[0]
            print(f"  - Loaded Grid Name: {loaded_grid.name}")
            print(f"  - Loaded Grid Class: {loaded_grid.gridClass}")
            print(f"  - Loaded Active Voxels: {loaded_grid.activeVoxelCount()}")

            # Verification
            if loaded_grid.activeVoxelCount() == first_vdb_grid.activeVoxelCount():
                print("  - VDB Save/Load cycle VERIFIED successfully.")
            else:
                print("  - VDB Save/Load cycle FAILED.")

    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # --- Clean up ---
        print("\nCleaning up test directory...")
        for f in test_dir.iterdir():
            f.unlink()
        test_dir.rmdir()
        print("Cleanup complete.")
