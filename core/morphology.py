# -*- coding: utf-8 -*-
"""
Module: morphology.py
Author: Gemini
Description: Provides a suite of 3D GRAYSCALE morphological operations for
             processing voxel data. This allows for nuanced smoothing and
             refinement of gradient maps.
"""

import numpy as np
from scipy import ndimage
from typing import Optional

def apply_3d_erosion(
    voxel_window: np.ndarray,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Performs 3D grayscale erosion on a voxel window. This darkens areas and
    is useful for diminishing weaker contours.

    Args:
        voxel_window (np.ndarray): The input 3D array (grayscale uint8).
        iterations (int): The number of times to apply the erosion.
        structure (Optional[np.ndarray]): The structuring element. If None,
                                           a 3x3x3 cube is used.

    Returns:
        np.ndarray: The eroded grayscale voxel window.
    """
    # Use the grayscale version of the function
    return ndimage.grey_erosion(
        voxel_window,
        footprint=structure, # Note: SciPy uses 'footprint' for grayscale
        size=None if structure is not None else (3,3,3) # or 'size' for default
    ).astype(np.uint8)


def apply_3d_dilation(
    voxel_window: np.ndarray,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Performs 3D grayscale dilation on a voxel window. This brightens areas
    and is useful for enhancing contours.

    Args:
        voxel_window (np.ndarray): The input 3D array (grayscale uint8).
        iterations (int): The number of times to apply the dilation.
        structure (Optional[np.ndarray]): The structuring element. If None,
                                           a 3x3x3 cube is used.

    Returns:
        np.ndarray: The dilated grayscale voxel window.
    """
    # Use the grayscale version of the function
    return ndimage.grey_dilation(
        voxel_window,
        footprint=structure,
        size=None if structure is not None else (3,3,3)
    ).astype(np.uint8)


def apply_3d_opening(
    voxel_window: np.ndarray,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Performs 3D grayscale opening (erosion then dilation). This smooths bright
    peaks and removes small, bright noise.

    Args:
        voxel_window (np.ndarray): The input 3D array (grayscale uint8).
        iterations (int): The number of times to apply the operation.
        structure (Optional[np.ndarray]): The structuring element. If None,
                                           a 3x3x3 cube is used.

    Returns:
        np.ndarray: The opened grayscale voxel window.
    """
    # Use the grayscale version of the function
    return ndimage.grey_opening(
        voxel_window,
        footprint=structure,
        size=None if structure is not None else (3,3,3)
    ).astype(np.uint8)


def apply_3d_closing(
    voxel_window: np.ndarray,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Performs 3D grayscale closing (dilation then erosion). This fills in
    small dark areas and valleys.

    Args:
        voxel_window (np.ndarray): The input 3D array (grayscale uint8).
        iterations (int): The number of times to apply the operation.
        structure (Optional[np.ndarray]): The structuring element. If None,
                                           a 3x3x3 cube is used.

    Returns:
        np.ndarray: The closed grayscale voxel window.
    """
    # Use the grayscale version of the function
    return ndimage.grey_closing(
        voxel_window,
        footprint=structure,
        size=None if structure is not None else (3,3,3)
    ).astype(np.uint8)


# --- Example Usage ---
if __name__ == '__main__':
    import cv2
    from pathlib import Path
    from core.operators_3d import apply_3d_gradient

    print("--- 3D Grayscale Morphology Test ---")

    # --- Create a synthetic 3D object (a sphere) ---
    D, H, W = 64, 128, 128
    center_d, center_h, center_w = D // 2, H // 2, W // 2
    radius = D // 3
    z, y, x = np.ogrid[:D, :H, :W]
    sphere_mask = (x - center_w)**2 + (y - center_h)**2 + (z - center_d)**2 <= radius**2
    test_volume = np.zeros((D, H, W), dtype=np.uint8)
    test_volume[sphere_mask] = 255
    
    # --- First, create a gradient map to work on ---
    gradient_map = apply_3d_gradient(test_volume, operator='scharr')
    print(f"Created a gradient map of shape {gradient_map.shape}")

    # --- Apply grayscale morphological operations ---
    print("Applying grayscale operations...")
    opened_volume = apply_3d_opening(gradient_map)
    closed_volume = apply_3d_closing(gradient_map)
    
    print("Operations complete.")

    # --- Save results for visual inspection ---
    output_dir = Path("./grayscale_morphology_test_output")
    output_dir.mkdir(exist_ok=True)
    center_slice_idx = D // 2

    def save_slice(volume, name):
        path = str(output_dir / f"{name}.png")
        cv2.imwrite(path, volume[center_slice_idx])

    save_slice(gradient_map, "00_input_gradient_map")
    save_slice(opened_volume, "01_opened_gradient")   # Should look smoother
    save_slice(closed_volume, "02_closed_gradient")   # Should look bolder/filled in

    print(f"\nTest complete. Output images saved in '{output_dir.resolve()}'")
    print("Inspect the images. The results should be subtle grayscale changes, not binary shapes.")
