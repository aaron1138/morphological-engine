# -*- coding: utf-8 -*-
"""
Module: dask_handler.py
Author: Jules
Description: Core functions for creating and managing Dask arrays from image data.
"""

import os
import dask
import dask.array as da
import numpy as np
from PIL import Image
from typing import Tuple, List

def load_image_stack_as_dask_array(directory: str, chunk_size: Tuple[int, int, int]) -> da.Array:
    """
    Lazily loads a stack of PNG images from a directory into a 3D Dask array.

    Args:
        directory: The path to the directory containing the PNG image stack.
        chunk_size: The desired chunk size for the Dask array (depth, height, width).

    Returns:
        A 3D Dask array representing the image volume. Returns None if no images are found.
    """
    print(f"Scanning directory for PNGs: {directory}")

    # Find and sort all PNG files to ensure correct Z-order
    try:
        image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.png')])
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory}")
        return None

    if not image_files:
        print("No PNG files found in the specified directory.")
        return None

    print(f"Found {len(image_files)} PNG files.")

    # Lazily load each image using dask.delayed
    @dask.delayed
    def load_image(path):
        with Image.open(path) as img:
            # Ensure image is converted to 8-bit grayscale ('L')
            return np.array(img.convert('L'))

    # Get the shape from the first image to determine the dimensions
    try:
        sample_img = load_image(image_files[0]).compute()
        height, width = sample_img.shape
        depth = len(image_files)
        print(f"Image stack dimensions (DxHxW): {depth}x{height}x{width}")
    except Exception as e:
        print(f"Error reading sample image to determine dimensions: {e}")
        return None

    # Create a list of delayed objects
    lazy_images = [load_image(path) for path in image_files]

    # Stack the delayed objects into a 3D Dask array
    # Each image becomes a slice along the first (Z) axis
    dask_array = da.stack(lazy_images, axis=0)

    # Re-chunk the array to the desired 3D chunking scheme
    print(f"Re-chunking array to {chunk_size}...")
    dask_array = dask_array.rechunk(chunks=chunk_size)

    print("Dask array created successfully.")
    return dask_array
