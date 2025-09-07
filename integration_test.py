import numpy as np
from pathlib import Path
import sys

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.voxel_engine import VoxelEngine
from core.slice_loader import SliceLoader
from data_io.image import write_image
from gpu.gpu_context import GPUContext

def create_dummy_image_sequence(directory: Path, count: int, shape: tuple):
    """Helper function to create a sequence of PNG files."""
    print(f"\n--- Creating {count} dummy images in {directory} ---")
    for i in range(count):
        filepath = directory / f"slice_{i:04d}.png"
        # Create a simple image with a gradient that changes with each slice
        data = np.zeros(shape, dtype=np.uint8)
        gradient = np.linspace(0, 255, shape[1], dtype=np.uint8)
        data[:, :] = gradient * ((i + 1) / count)
        write_image(filepath, data.astype(np.uint8))
    print("--- Dummy images created ---")

def main():
    print("--- Starting Integration Test ---")
    test_dir = Path("./temp_integration_test_dir")

    try:
        # --- Setup ---
        image_count = 10
        image_shape = (64, 64)
        if test_dir.exists():
            # Clean up previous failed run if necessary
            for f in test_dir.glob('*.png'):
                f.unlink()
            test_dir.rmdir()
        test_dir.mkdir(exist_ok=True)
        create_dummy_image_sequence(test_dir, image_count, image_shape)

        # --- Step 1: Load image sequence and create VDB grid ---
        print("\n--- Testing Core Voxel Engine ---")
        loader = SliceLoader(str(test_dir))
        engine = VoxelEngine(loader, window_size=5)

        # Check if pyopenvdb is available before proceeding
        if 'vdb' not in sys.modules or sys.modules['vdb'] is None:
            print("WARNING: pyopenvdb is not installed. Skipping VDB grid test.")
        else:
            vdb_grid = next(engine.iter_vdb_grids())
            if vdb_grid and vdb_grid.activeVoxelCount() > 0:
                print("SUCCESS: VoxelEngine created an OpenVDB grid.")
                print(f"  - Grid Name: {vdb_grid.name}")
                print(f"  - Active Voxels: {vdb_grid.activeVoxelCount()}")
            else:
                raise RuntimeError("VoxelEngine did not create a valid OpenVDB grid.")

        # --- Step 2: Initialize GPU Context ---
        print("\n--- Testing GPU Context Initialization ---")
        gpu = GPUContext()
        if gpu.ctx:
            print("SUCCESS: GPUContext initialized successfully.")
        else:
            raise RuntimeError("GPUContext could not be initialized.")

        # --- Step 3: (Placeholder) GPU Data Transfer ---
        print("\n--- Testing GPU Data Transfer (Placeholder) ---")
        # In the future, we would convert NanoVDB data and transfer it.
        # For now, we just confirm we can create resources.
        try:
            tex = gpu.create_texture(image_shape)
            print(f"SUCCESS: Placeholder test - Created a GPU texture of size {image_shape}.")
            tex.release()
        except Exception as e:
            raise RuntimeError(f"Placeholder test - Could not create GPU texture. Error: {e}")

        gpu.release()

        print("\n-----------------------------")
        print("--- Integration Test PASSED ---")
        print("-----------------------------")

    except Exception as e:
        print(f"\n--- Integration Test FAILED ---")
        print(f"An error occurred: {e}")
        # Exit with a non-zero code to indicate failure, useful for CI
        sys.exit(1)
    finally:
        # --- Cleanup ---
        print("\n--- Cleaning up test environment ---")
        if test_dir.exists():
            for f in test_dir.glob('*.png'):
                f.unlink()
            test_dir.rmdir()
        print("--- Cleanup complete ---")

if __name__ == '__main__':
    main()
