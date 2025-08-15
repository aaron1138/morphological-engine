# -*- coding: utf-8 -*-
"""
Module: operators_3d.py
Author: Gemini
Description: Implements 3D gradient operators (Sobel, Scharr) for creating
             contour maps from 3D voxel data. This module prioritizes
             correctness, symmetry, and safe data type handling.
"""

import numpy as np
from scipy import ndimage
from typing import Literal

# --- 3D Sobel Operator Kernels ---
# These are 3x3x3 kernels. The axis indicates the direction of the gradient.
SOBEL_Z = np.array([
    [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
], dtype=np.int32)

SOBEL_Y = SOBEL_Z.transpose((1, 0, 2)) # Swap Z and Y axes
SOBEL_X = SOBEL_Z.transpose((2, 1, 0)) # Swap Z and X axes

# --- 3D Scharr Operator Kernels (Higher quality rotational symmetry) ---
SCHARR_Z = np.array([
    [[3, 10, 3], [10, 32, 10], [3, 10, 3]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[-3, -10, -3], [-10, -32, -10], [-3, -10, -3]]
], dtype=np.int32)

SCHARR_Y = SCHARR_Z.transpose((1, 0, 2)) # Swap Z and Y axes
SCHARR_X = SCHARR_Z.transpose((2, 1, 0)) # Swap Z and X axes

# --- Main Function ---

def apply_3d_gradient(
    voxel_window: np.ndarray,
    operator: Literal['sobel', 'scharr'] = 'sobel',
    output_dtype: np.dtype = np.uint8
) -> np.ndarray:
    """
    Applies a 3D gradient operator to a voxel window to generate a contour map.

    Args:
        voxel_window (np.ndarray): The input 3D NumPy array (depth, height, width).
                                   Expected to be unsigned integer type (e.g., uint8).
        operator (Literal['sobel', 'scharr']): The gradient operator to use.
        output_dtype (np.dtype): The desired NumPy data type for the output array.
                                 Typically np.uint8 for image output.

    Returns:
        np.ndarray: A 3D NumPy array of the same shape as the input, containing
                    the gradient magnitude map, scaled to the range of `output_dtype`.
    """
    if voxel_window.ndim != 3:
        raise ValueError("Input `voxel_window` must be a 3D array.")

    # --- Select Kernels ---
    if operator == 'sobel':
        kernel_x, kernel_y, kernel_z = SOBEL_X, SOBEL_Y, SOBEL_Z
    elif operator == 'scharr':
        kernel_x, kernel_y, kernel_z = SCHARR_X, SCHARR_Y, SCHARR_Z
    else:
        raise ValueError("Operator must be 'sobel' or 'scharr'")

    # --- Perform Convolution ---
    # Cast input to a larger, signed integer type to prevent overflow.
    data = voxel_window.astype(np.int32)

    grad_x = ndimage.convolve(data, kernel_x, mode='constant', cval=0.0)
    grad_y = ndimage.convolve(data, kernel_y, mode='constant', cval=0.0)
    grad_z = ndimage.convolve(data, kernel_z, mode='constant', cval=0.0)

    # --- Calculate Gradient Magnitude ---
    # *** FIX: Explicitly cast gradients to float64 before squaring and taking ***
    # *** the square root. This prevents the TypeError with the ufunc.      ***
    grad_x_f = grad_x.astype(np.float64)
    grad_y_f = grad_y.astype(np.float64)
    grad_z_f = grad_z.astype(np.float64)

    magnitude = np.sqrt(grad_x_f**2 + grad_y_f**2 + grad_z_f**2)

    # --- Scale Result to Output Data Type ---
    min_mag, max_mag = magnitude.min(), magnitude.max()
    
    if max_mag > min_mag:
        # Avoid division by zero if the magnitude is flat
        magnitude = (magnitude - min_mag) / (max_mag - min_mag)
    
    if np.issubdtype(output_dtype, np.integer):
        max_val = np.iinfo(output_dtype).max
        magnitude = (magnitude * max_val)
    
    return magnitude.astype(output_dtype)


# --- Example Usage ---
if __name__ == '__main__':
    import cv2
    from pathlib import Path

    print("--- 3D Gradient Operator Test ---")

    # --- Create a synthetic 3D object (a sphere) for testing ---
    D, H, W = 64, 128, 128  # Depth, Height, Width
    center_d, center_h, center_w = D // 2, H // 2, W // 2
    radius = D // 3

    z, y, x = np.ogrid[:D, :H, :W]
    
    # Equation of a sphere
    sphere_mask = (x - center_w)**2 + (y - center_h)**2 + (z - center_d)**2 <= radius**2
    
    # Create a uint8 voxel window with the sphere
    test_volume = np.zeros((D, H, W), dtype=np.uint8)
    test_volume[sphere_mask] = 255
    
    print(f"Created a test volume of shape {test_volume.shape} with a sphere.")

    # --- Apply the Sobel operator ---
    print("Applying 3D Sobel operator...")
    sobel_magnitude = apply_3d_gradient(test_volume, operator='sobel', output_dtype=np.uint8)

    # --- Apply the Scharr operator ---
    print("Applying 3D Scharr operator...")
    scharr_magnitude = apply_3d_gradient(test_volume, operator='scharr', output_dtype=np.uint8)

    # --- Save the central slice of the results for visual inspection ---
    output_dir = Path("./operator_test_output")
    output_dir.mkdir(exist_ok=True)

    # The middle slice along the Z-axis should show a perfect circle contour
    center_slice_idx = D // 2
    
    input_slice_path = str(output_dir / "00_input_sphere_slice.png")
    sobel_slice_path = str(output_dir / "01_sobel_magnitude_slice.png")
    scharr_slice_path = str(output_dir / "02_scharr_magnitude_slice.png")

    cv2.imwrite(input_slice_path, test_volume[center_slice_idx])
    cv2.imwrite(sobel_slice_path, sobel_magnitude[center_slice_idx])
    cv2.imwrite(scharr_slice_path, scharr_magnitude[center_slice_idx])

    print(f"\nTest complete. Output images saved in '{output_dir.resolve()}'")
    print("Inspect the output images. They should show a clean, symmetrical circle.")

