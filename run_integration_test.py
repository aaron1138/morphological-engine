"""
Main script for the Voxel Processing Engine.

This script serves as an end-to-end test and demonstration of the foundational
modules: vdb_handler and gpu_processor.

It performs the following steps:
1. Creates a sample OpenVDB sphere grid and saves it.
2. Initializes a headless GPU context using the GPUProcessor.
3. Simulates loading a 2D slice of data into a GPU texture.
4. Loads and runs a simple pass-through shader to process the texture.
5. Reads the output from the GPU and saves it as a PNG image.
"""

import os
import numpy as np
from PIL import Image

from src.vdb_handler import (
    create_level_set_sphere,
    write_grid,
    read_grid,
    convert_to_nanovdb
)
from src.gpu_processor import GPUProcessor

# Define the workspace directory for our outputs
WORKSPACE_DIR = 'workspace'
SHADERS_DIR = 'shaders'


def create_synthetic_slice(size: tuple) -> np.ndarray:
    """
    Creates a NumPy array representing a grayscale slice of a sphere.
    This simulates reading a 2D slice from a VDB grid.
    """
    width, height = size
    # Create a coordinate grid
    y, x = np.ogrid[-height/2:height/2, -width/2:width/2]

    radius = min(width, height) * 0.4

    # Create a circle mask
    mask = x*x + y*y <= radius*radius

    # Create a grayscale image (8-bit unsigned integer)
    # The image should be (height, width, 1) to match texture expectations
    image = np.zeros((height, width, 1), dtype=np.uint8)
    image[mask] = 255  # White circle on black background

    return image


def main():
    """Main execution function."""
    print("--- Voxel Processing Engine: Integration Test ---")

    # Ensure the workspace directory exists
    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR)

    # --- 1. VDB Handler Demonstration ---
    print("\n--- Step 1: Testing VDB Handler ---")
    vdb_output_path = os.path.join(WORKSPACE_DIR, 'test_sphere.vdb')

    # Create a sample VDB grid
    # Note: This uses the dummy module if pyopenvdb is not installed.
    sphere_grid = create_level_set_sphere(radius=25.0)

    # Write the grid to a file
    write_grid(vdb_output_path, sphere_grid)

    # Read the grid back (verifies the read function signature)
    read_grid_obj = read_grid(vdb_output_path)
    if read_grid_obj:
        print(f"Successfully read back grid with name: {read_grid_obj.name}")

    # Test the NanoVDB conversion placeholder
    nanovdb_buffer = convert_to_nanovdb(sphere_grid)
    print(f"NanoVDB conversion placeholder returned buffer of size: {len(nanovdb_buffer)} bytes")

    # --- 2. GPU Processor and Rendering Pipeline ---
    print("\n--- Step 2: Testing GPU Processor ---")

    image_size = (512, 512)
    png_output_path = os.path.join(WORKSPACE_DIR, 'output.png')

    # Create synthetic data to simulate loading from VDB
    input_slice_numpy = create_synthetic_slice(image_size)

    # Save the synthetic input for comparison
    Image.fromarray(input_slice_numpy.squeeze(), 'L').save(os.path.join(WORKSPACE_DIR, 'input_synthetic.png'))
    print(f"Saved synthetic input slice to {os.path.join(WORKSPACE_DIR, 'input_synthetic.png')}")

    # Initialize the GPU processor
    gpu = GPUProcessor()
    if not gpu.ctx:
        print("Failed to initialize GPUProcessor. Aborting.")
        return

    # Load the pass-through shaders
    vert_shader = os.path.join(SHADERS_DIR, 'passthrough.vert')
    frag_shader = os.path.join(SHADERS_DIR, 'passthrough.frag')
    gpu.load_shader_program(vert_shader, frag_shader)

    # Create the GPU resources
    input_texture = gpu.ctx.texture(image_size, components=1, data=input_slice_numpy.tobytes(), dtype='f1')
    target_fbo, color_texture = gpu.create_offscreen_buffer(image_size, components=1, dtype='f1')
    gpu.create_fullscreen_quad()

    # Run the render job
    print("Running render job...")
    gpu.render(target_fbo=target_fbo, texture_to_process=input_texture)

    # Save the output from the framebuffer to a PNG
    gpu.save_output_to_png(target_fbo, png_output_path)

    # --- 3. Cleanup ---
    print("\n--- Step 3: Cleanup ---")
    # Release GPU resources
    input_texture.release()
    target_fbo.release()
    color_texture.release()
    gpu.destroy()

    print("\n--- Integration Test Complete ---")
    print(f"Check the '{WORKSPACE_DIR}' directory for output files.")


if __name__ == '__main__':
    main()
