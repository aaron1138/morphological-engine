import dask
import dask.array as da
import imageio.v2 as imageio
import os
import re
import numpy as np

def _sorted_image_files(input_dir: str) -> list[str]:
    """
    Finds all PNG files in a directory and sorts them numerically.
    """
    numeric_pattern = re.compile(r'(\d+)\.png$')

    def get_numeric_part(filename):
        match = numeric_pattern.search(filename.lower())
        return int(match.group(1)) if match else float('inf')

    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    sorted_files = sorted(all_files, key=get_numeric_part)

    if not sorted_files:
        raise ValueError(f"No PNG files found in directory: {input_dir}")

    return [os.path.join(input_dir, f) for f in sorted_files]

def extract_orthogonal_slices(input_dir: str, output_dir: str, chunk_size: tuple, progress_callback=None):
    """
    Loads a stack of PNGs, builds a 3D Dask array, and extracts orthogonal XZ and YZ slices.

    Args:
        input_dir (str): Directory containing the input PNG sequence (XY planes).
        output_dir (str): Directory where the output slices will be saved.
        chunk_size (tuple): The desired (Z, Y, X) chunk size for the Dask array.
        progress_callback (function, optional): A function to call with progress updates.
                                                It receives (current_step, total_steps, message).
    """
    print(f"Starting orthogonal slice extraction for {input_dir}")

    # 1. Get sorted list of image files (these are our Z slices)
    image_files = _sorted_image_files(input_dir)

    # 2. Lazily load each image using dask.delayed
    # Assuming all images have the same shape. We read the first one to get dims.
    sample_image = imageio.imread(image_files[0])
    if sample_image.ndim == 3: # Convert RGB to grayscale
        sample_image = sample_image[:, :, 0]

    lazy_images = [dask.delayed(imageio.imread)(f) for f in image_files]

    # 3. Create the 3D Dask array
    # We need to handle both grayscale (2D) and color (3D) images
    if sample_image.ndim == 2: # Grayscale
        stack = [da.from_delayed(img, shape=sample_image.shape, dtype=sample_image.dtype) for img in lazy_images]
        dask_stack = da.stack(stack, axis=0)
    else: # Color - take the first channel for now
         stack = [da.from_delayed(img, shape=sample_image.shape, dtype=sample_image.dtype)[:,:,0] for img in lazy_images]
         dask_stack = da.stack(stack, axis=0)

    print(f"Original Dask array created with shape: {dask_stack.shape} and chunks: {dask_stack.chunksize}")

    # 4. Re-chunk the array into volumetric blocks
    # Dask chunks are (Z, Y, X)
    volumetric_stack = dask_stack.rechunk(chunks=chunk_size)
    print(f"Re-chunked to volumetric chunks: {volumetric_stack.chunksize}")

    # --- 5. Extract and Save Slices ---
    os.makedirs(os.path.join(output_dir, "xz_slices"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "yz_slices"), exist_ok=True)

    total_steps = volumetric_stack.shape[1] + volumetric_stack.shape[2]
    current_step = 0

    # Extract XZ slices (iterating along the Y axis)
    print("Extracting XZ slices...")
    for y in range(volumetric_stack.shape[1]):
        if progress_callback:
            progress_callback(current_step, total_steps, f"Extracting XZ slice {y+1}/{volumetric_stack.shape[1]}")

        xz_slice = volumetric_stack[:, y, :].compute()
        # Transpose so Z is vertical and X is horizontal
        imageio.imwrite(os.path.join(output_dir, "xz_slices", f"slice_{y:04d}.png"), xz_slice.T)
        current_step += 1

    # Extract YZ slices (iterating along the X axis)
    print("Extracting YZ slices...")
    for x in range(volumetric_stack.shape[2]):
        if progress_callback:
            progress_callback(current_step, total_steps, f"Extracting YZ slice {x+1}/{volumetric_stack.shape[2]}")

        yz_slice = volumetric_stack[:, :, x].compute()
        # Transpose so Z is vertical and Y is horizontal
        imageio.imwrite(os.path.join(output_dir, "yz_slices", f"slice_{x:04d}.png"), yz_slice.T)
        current_step += 1

    if progress_callback:
        progress_callback(total_steps, total_steps, "Extraction complete.")

    print("Orthogonal slice extraction finished.")
