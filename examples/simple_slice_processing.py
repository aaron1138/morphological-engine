import sys
import os
import time
import numpy as np
from PIL import Image

# Add the src directory to the Python path to import the utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.vdb_utils import create_sphere_grid
from src.gl_utils import HeadlessRenderer, get_invert_shader
from src.processing import sample_grid_slice_python, sample_grid_slice_numba

def main():
    """
    Main function to run the proof-of-concept example.
    """
    print("--- Starting Proof-of-Concept Example ---")

    # --- Part 1: Voxel Data Creation ---
    print("\nStep 1: Creating a voxel grid...")
    sphere_grid = create_sphere_grid(radius=40.0, voxel_size=1.0)
    print(f"Created sphere grid with {sphere_grid.activeVoxelCount()} active voxels.")
    min_val, max_val = sphere_grid.minMaxValues()

    # --- Part 2: Slicing the Voxel Grid ---
    print("\nStep 2: Sampling a 2D slice from the grid...")
    slice_width, slice_height = 256, 256
    slice_z, center_x, center_y = 0, 0, 0

    # Use the pure Python version for the main workflow
    start_time = time.time()
    slice_numpy = sample_grid_slice_python(sphere_grid, slice_z, slice_width, slice_height, center_x, center_y)
    duration_py = time.time() - start_time
    print(f"Sampled a {slice_width}x{slice_height} slice using pure Python in {duration_py:.4f} seconds.")

    # --- Numba Demonstration ---
    # The Numba-jitted function is called here to demonstrate its use.
    # NOTE: As explained in processing.py, the Numba function uses a simulated
    # value because it cannot access the OpenVDB accessor object directly.
    # This call serves to demonstrate the integration pattern.
    print("\nDemonstrating Numba JIT compilation...")
    # First call will be slow due to compilation
    start_time = time.time()
    _ = sample_grid_slice_numba(None, min_val, max_val, slice_z, slice_width, slice_height, center_x, center_y)
    duration_jit_cold = time.time() - start_time
    print(f"Numba JIT (cold start) took: {duration_jit_cold:.4f} seconds.")
    # Second call will be fast
    start_time = time.time()
    _ = sample_grid_slice_numba(None, min_val, max_val, slice_z, slice_width, slice_height, center_x, center_y)
    duration_jit_warm = time.time() - start_time
    print(f"Numba JIT (warm start) took: {duration_jit_warm:.4f} seconds.")


    # Save the raw slice for inspection
    raw_slice_img = Image.fromarray((slice_numpy * 255).astype(np.uint8), 'L')
    raw_slice_img.save("raw_slice.png")
    print("\nSaved raw slice to raw_slice.png")

    # --- Part 3: GPU Processing with ModernGL ---
    print("\nStep 3: Processing the slice on the GPU...")

    slice_rgb = np.stack([slice_numpy] * 3, axis=-1)
    renderer = None
    try:
        shaders = get_invert_shader()
        renderer = HeadlessRenderer(slice_width, slice_height, shaders)

        processed_slice_rgb = renderer.process_image(slice_rgb)
        print("GPU processing complete.")

        processed_slice_gray = processed_slice_rgb.mean(axis=2)
        processed_slice_img = Image.fromarray((processed_slice_gray * 255).astype(np.uint8), 'L')
        processed_slice_img.save("processed_slice.png")
        print("Saved processed slice to processed_slice.png")

    except Exception as e:
        print(f"\nAn error occurred during GPU processing: {e}")
        print("This may be because the required libraries (OpenGL, etc.) are not available in the current environment.")

    finally:
        if renderer:
            renderer.release()

    print("\n--- Proof-of-Concept Example Finished ---")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"ImportError: {e}")
        print("\nPlease ensure you have set up the environment correctly as described in docs/SETUP.md")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
