import os
import dask.array as da
from dask import delayed
from PIL import Image
import numpy as np
from typing import Tuple

def load_png_stack_as_dask_array(directory: str, chunk_size: Tuple[int, int, int] = (64, 64, 64)):
    """
    Lazily loads a stack of PNG images from a directory into a 3D Dask array.

    The images are assumed to be sorted alphabetically and represent slices
    along the Z-axis of a 3D volume.

    Args:
        directory (str): The path to the directory containing the PNG files.
        chunk_size (Tuple[int, int, int]): The desired chunk size for the Dask array (Z, Y, X).

    Returns:
        dask.array.Array: A 3D Dask array representing the image stack, or None if
                          the directory is empty or contains no PNGs.
    """
    print(f"Loading PNG stack from directory: {directory}")

    # Find and sort all PNG files
    try:
        filenames = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.png')])
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory}")
        return None

    if not filenames:
        print("Error: No PNG files found in the specified directory.")
        return None

    # Get shape and dtype from the first image
    with Image.open(filenames[0]) as img:
        sample_img = np.array(img)
        # Assuming 8-bit grayscale images
        if sample_img.ndim != 2:
            print(f"Warning: First image is not 2D grayscale. Shape is {sample_img.shape}. Taking first channel.")
            sample_img = sample_img[:, :, 0]

        height, width = sample_img.shape
        dtype = sample_img.dtype

    num_slices = len(filenames)
    shape = (num_slices, height, width)
    print(f"Detected volume shape: {shape} with dtype: {dtype}")

    # Create a delayed function to load a single image
    @delayed
    def load_image(filename):
        with Image.open(filename) as img:
            img_array = np.array(img)
            if img_array.ndim != 2:
                return img_array[:, :, 0]
            return img_array

    # Create a list of delayed loaders
    delayed_stack = [load_image(f) for f in filenames]

    # Create a Dask array from the stack of delayed objects
    # We create a stack of 2D arrays, then reshape to 3D
    dask_stack = [da.from_delayed(d, shape=(height, width), dtype=dtype) for d in delayed_stack]

    # Stack them along the first (Z) axis
    dask_array = da.stack(dask_stack, axis=0)

    # Rechunk to the desired 3D chunk size
    final_array = dask_array.rechunk(chunks=chunk_size)

    print(f"Successfully created Dask array with chunks: {final_array.chunksize}")
    return final_array


if __name__ == '__main__':
    # This example demonstrates how to use the image loader.
    # It requires a directory with some sample PNGs to run.

    print("--- Running Image Loader Example ---")

    # Create a dummy directory with some fake images for demonstration
    dummy_dir = "temp_png_stack"
    if not os.path.exists(dummy_dir):
        os.makedirs(dummy_dir)

    print(f"Creating dummy PNG files in '{dummy_dir}'...")
    for i in range(10):
        # Create a 128x128 grayscale image
        img_data = np.full((128, 128), i * 25, dtype=np.uint8)
        img = Image.fromarray(img_data, 'L')
        img.save(os.path.join(dummy_dir, f"slice_{i:03d}.png"))

    # Load the dummy stack as a Dask array
    dask_volume = load_png_stack_as_dask_array(dummy_dir, chunk_size=(3, 64, 64))

    if dask_volume is not None:
        print("\n--- Dask Array Properties ---")
        print(f"Shape: {dask_volume.shape}")
        print(f"Chunk Size: {dask_volume.chunksize}")
        print(f"Data Type: {dask_volume.dtype}")

        # To prove it's lazy, we'll compute the mean of the whole volume
        print("\nComputing mean of the volume (this will trigger the actual loading)...")
        mean_value = dask_volume.mean().compute()
        print(f"Computed mean value: {mean_value}")

    # Clean up the dummy files
    print("\nCleaning up dummy files...")
    for f in os.listdir(dummy_dir):
        os.remove(os.path.join(dummy_dir, f))
    os.rmdir(dummy_dir)

    print("\n--- Image Loader Example Finished ---")
