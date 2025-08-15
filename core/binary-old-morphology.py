# -*- coding: utf-8 -*-
"""
Module: morphology.py
Author: Gemini
Description: Provides a suite of 3D morphological operations for processing
             voxel data. This includes erosion, dilation, opening, and closing,
             which are essential for refining feature geometry.
"""

import numpy as np
from scipy import ndimage
from typing import Optional

def _prepare_binary_input(
    voxel_window: np.ndarray,
    threshold_value: int = 0  # <<< FIX: Changed default threshold from 127 to 0
) -> np.ndarray:
    """
    Converts an integer voxel window into a binary (boolean) array.
    
    Args:
        voxel_window (np.ndarray): Input data, typically uint8.
        threshold_value (int): Pixel value above which voxels are considered 'True'.
                               A value of 0 means any non-black pixel is True.

    Returns:
        np.ndarray: A boolean array of the same shape as the input.
    """
    if voxel_window.dtype == bool:
        return voxel_window # Already binary
    
    return voxel_window > threshold_value

def apply_3d_erosion(
    voxel_window: np.ndarray,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Performs 3D erosion on a voxel window. This shrinks bright areas.

    Args:
        voxel_window (np.ndarray): The input 3D array (binary or uint8).
        iterations (int): The number of times to apply the erosion.
        structure (Optional[np.ndarray]): The structuring element. If None,
                                           a 3x3x3 cube is used.

    Returns:
        np.ndarray: The eroded voxel window as a uint8 array (0 or 255).
    """
    binary_input = _prepare_binary_input(voxel_window)
    
    eroded_binary = ndimage.binary_erosion(
        binary_input, 
        structure=structure, 
        iterations=iterations
    )
    
    return eroded_binary.astype(np.uint8) * 255

def apply_3d_dilation(
    voxel_window: np.ndarray,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Performs 3D dilation on a voxel window. This expands bright areas.

    Args:
        voxel_window (np.ndarray): The input 3D array (binary or uint8).
        iterations (int): The number of times to apply the dilation.
        structure (Optional[np.ndarray]): The structuring element. If None,
                                           a 3x3x3 cube is used.

    Returns:
        np.ndarray: The dilated voxel window as a uint8 array (0 or 255).
    """
    binary_input = _prepare_binary_input(voxel_window)
    
    dilated_binary = ndimage.binary_dilation(
        binary_input, 
        structure=structure, 
        iterations=iterations
    )
    
    return dilated_binary.astype(np.uint8) * 255

def apply_3d_opening(
    voxel_window: np.ndarray,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Performs 3D opening (erosion then dilation). Removes small bright spots.

    Args:
        voxel_window (np.ndarray): The input 3D array (binary or uint8).
        iterations (int): The number of times to apply the operation.
        structure (Optional[np.ndarray]): The structuring element. If None,
                                           a 3x3x3 cube is used.

    Returns:
        np.ndarray: The opened voxel window as a uint8 array (0 or 255).
    """
    binary_input = _prepare_binary_input(voxel_window)
    
    opened_binary = ndimage.binary_opening(
        binary_input, 
        structure=structure, 
        iterations=iterations
    )
    
    return opened_binary.astype(np.uint8) * 255

def apply_3d_closing(
    voxel_window: np.ndarray,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Performs 3D closing (dilation then erosion). Fills small dark holes.

    Args:
        voxel_window (np.ndarray): The input 3D array (binary or uint8).
        iterations (int): The number of times to apply the operation.
        structure (Optional[np.ndarray]): The structuring element. If None,
                                           a 3x3x3 cube is used.

    Returns:
        np.ndarray: The closed voxel window as a uint8 array (0 or 255).
    """
    binary_input = _prepare_binary_input(voxel_window)
    
    closed_binary = ndimage.binary_closing(
        binary_input, 
        structure=structure, 
        iterations=iterations
    )
    
    return closed_binary.astype(np.uint8) * 255


# --- Example Usage ---
if __name__ == '__main__':
    import cv2
    from pathlib import Path

    print("--- 3D Morphology Test ---")

    # --- Create a synthetic 3D object (a sphere with a hole and noise) ---
    D, H, W = 64, 128, 128
    center_d, center_h, center_w = D // 2, H // 2, W // 2
    radius = D // 3
    
    z, y, x = np.ogrid[:D, :H, :W]
    
    # Create a base sphere
    sphere_mask = (x - center_w)**2 + (y - center_h)**2 + (z - center_d)**2 <= radius**2
    test_volume = np.zeros((D, H, W), dtype=np.uint8)
    test_volume[sphere_mask] = 255
    
    # Add noise (small bright spots outside)
    noise = np.random.rand(D, H, W) > 0.995
    test_volume[noise] = 255
    
    # Add a hole (small dark spot inside)
    hole_mask = (x - center_w)**2 + (y - center_h)**2 + (z - (center_d - 5))**2 <= (radius//4)**2
    test_volume[hole_mask] = 0
    
    print(f"Created a test volume of shape {test_volume.shape} with a noisy sphere containing a hole.")

    # --- Set parameters ---
    ITERATIONS = 2

    # --- Apply morphological operations ---
    print(f"Applying operations with {ITERATIONS} iterations...")
    eroded_volume = apply_3d_erosion(test_volume, iterations=ITERATIONS)
    dilated_volume = apply_3d_dilation(test_volume, iterations=ITERATIONS)
    opened_volume = apply_3d_opening(test_volume, iterations=ITERATIONS)
    closed_volume = apply_3d_closing(test_volume, iterations=ITERATIONS)
    
    print("Operations complete.")

    # --- Save results for visual inspection ---
    output_dir = Path("./morphology_test_output")
    output_dir.mkdir(exist_ok=True)
    center_slice_idx = D // 2

    def save_slice(volume, name):
        path = str(output_dir / f"{name}.png")
        cv2.imwrite(path, volume[center_slice_idx])

    save_slice(test_volume, "00_input_noisy_sphere")
    save_slice(eroded_volume, "01_eroded_sphere")   # Should be smaller
    save_slice(dilated_volume, "02_dilated_sphere")  # Should be larger, hole smaller
    save_slice(opened_volume, "03_opened_sphere")   # Should have noise removed
    save_slice(closed_volume, "04_closed_sphere")   # Should have hole filled

    print(f"\nTest complete. Output images saved in '{output_dir.resolve()}'")
    print("Inspect the images to see the effects:")
    print(" - Erosion shrinks the sphere.")
    print(" - Dilation expands it.")
    print(" - Opening removes the external noise specks.")
    print(" - Closing fills the internal hole.")
