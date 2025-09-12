# core/engine.py

import os
import re
import dask
import dask.array as da
import numpy as np
import imageio.v2 as imageio
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import numba
import logging
import cv2
from utils import lut_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@numba.jit(nopython=True, cache=True)
def _calculate_edt_gradient_numba(distance_map, labels, num_labels, fade_distance_limit):
    """
    Calculates the Enhanced EDT gradient using a Numba JIT-compiled loop.
    This is adapted from the Voxel-Stack-Blender project.
    """
    final_gradient_map = np.zeros_like(distance_map, dtype=np.uint8)

    # First pass: find max distance for each label
    max_vals = np.zeros(num_labels, dtype=np.float32)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            if label > 0:
                dist = distance_map[y, x]
                if dist > max_vals[label]:
                    max_vals[label] = dist

    # Second pass: calculate final gradient
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            if label > 0:
                dist = distance_map[y, x]
                denominator = min(max_vals[label], fade_distance_limit)
                if denominator > 0:
                    clipped_dist = min(dist, denominator)
                    normalized = clipped_dist / denominator
                    inverted = 1.0 - normalized
                    final_gradient_map[y, x] = int(inverted * 255)

    return final_gradient_map

def _calculate_gradient_for_pass(current_mask, neighbor_masks, fade_distance_limit):
    """
    Core EDT logic for a single pass (either forward or backward).
    """
    if not neighbor_masks:
        return np.zeros_like(current_mask, dtype=np.uint8)

    # Combine all neighbor masks into one
    combined_neighbor_mask = np.zeros_like(current_mask, dtype=np.uint8)
    for mask in neighbor_masks:
        combined_neighbor_mask = np.maximum(combined_neighbor_mask, mask)

    # Identify the areas that exist in neighbors but not in the current slice
    transition_areas = cv2.bitwise_and(combined_neighbor_mask, cv2.bitwise_not(current_mask))
    if cv2.countNonZero(transition_areas) == 0:
        return np.zeros_like(current_mask, dtype=np.uint8)

    # Calculate distance from the edges of the current slice's features
    dist_transform_src = cv2.bitwise_not(current_mask)
    distance_map = cv2.distanceTransform(dist_transform_src, cv2.DIST_L2, 5)

    # We only care about the distances within the transition areas
    transition_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=transition_areas)

    # Label the distinct islands within the transition areas
    num_labels, labels = cv2.connectedComponents(transition_areas)
    if num_labels <= 1:
        return np.zeros_like(current_mask, dtype=np.uint8)

    # Use the Numba-accelerated function to calculate the gradient
    final_gradient = _calculate_edt_gradient_numba(
        transition_distance_map.astype(np.float32),
        labels.astype(np.int32),
        num_labels,
        fade_distance_limit
    )

    return cv2.bitwise_and(final_gradient, final_gradient, mask=transition_areas)


def _bidirectional_edt_on_chunk(chunk, look_backward, look_forward, fade_distance_limit):
    """
    This function is applied to each chunk of the Dask array via map_overlap.
    It calculates the bidirectional EDT for the main part of the chunk,
    using the "ghost" regions (overlap) as context.
    """
    num_slices, height, width = chunk.shape
    output_chunk = np.zeros((num_slices - look_backward - look_forward, height, width), dtype=np.uint8)

    # Convert the whole chunk to binary masks once
    binary_chunk = (chunk > 127).astype(np.uint8) * 255

    # Iterate only over the non-overlapping part of the chunk
    for i in range(look_backward, num_slices - look_forward):
        current_mask = binary_chunk[i]

        # Get backward and forward context slices
        backward_masks = [binary_chunk[j] for j in range(i - look_backward, i)]
        forward_masks = [binary_chunk[j] for j in range(i + 1, i + 1 + look_forward)]

        # Calculate gradients for both directions
        backward_gradient = _calculate_gradient_for_pass(current_mask, backward_masks, fade_distance_limit)
        forward_gradient = _calculate_gradient_for_pass(current_mask, forward_masks, fade_distance_limit)

        # Combine the gradients
        combined_gradient = np.maximum(backward_gradient, forward_gradient)

        # Merge with the original image data for the slice
        output_chunk[i - look_backward] = np.maximum(chunk[i], combined_gradient)

    return output_chunk

def apply_enhanced_edt(dask_stack, op, axis=0):
    """
    Applies the bidirectional Enhanced EDT to a Dask array along a specified axis.
    """
    logging.info(f"Applying Bidirectional Enhanced EDT along axis {axis}: look_backward={op.look_backward}, look_forward={op.look_forward}")

    # Define the depth of the overlap for the specified axis
    depth = {axis: (op.look_backward, op.look_forward), 0:0, 1:0, 2:0}
    depth[axis] = (op.look_backward, op.look_forward)

    # The function passed to map_overlap needs to know which axis to operate on.
    # We can't pass the axis directly, so we create a wrapper function.
    def edt_wrapper(chunk, look_backward, look_forward, fade_distance_limit):
        # Transpose the chunk so the target axis is always axis 0
        axes = list(range(chunk.ndim))
        axes.insert(0, axes.pop(axis))
        transposed_chunk = chunk.transpose(axes)

        # Run the original chunk-based function
        processed_transposed_chunk = _bidirectional_edt_on_chunk(
            transposed_chunk, look_backward, look_forward, fade_distance_limit
        )

        # Transpose the result back to the original orientation
        inverse_axes = [0] * chunk.ndim
        for i, p in enumerate(axes):
            inverse_axes[p] = i
        return processed_transposed_chunk.transpose(inverse_axes)

    result = dask_stack.map_overlap(
        edt_wrapper,
        depth=depth,
        boundary={0: 0, 1: 0, 2: 0},
        trim=False,
        look_backward=op.look_backward,
        look_forward=op.look_forward,
        fade_distance_limit=op.fade_distance_limit
    ).astype(np.uint8)

    return result

# --- Blend & Filter Operations ---

def apply_gaussian_blur(dask_array, op):
    """Applies Gaussian blur to a Dask array."""
    # For a 2D filter, we only need overlap on the spatial axes (1 and 2)
    # The overlap depth should be related to the kernel size
    k_size = op.gaussian_ksize_x # Assuming ksize is the same for x and y
    depth = int(k_size / 2)

    return dask_array.map_overlap(
        lambda block: cv2.GaussianBlur(block, (k_size, k_size), 0),
        depth={0: 0, 1: depth, 2: depth},
        boundary='reflect'
    ).astype(np.uint8)

@numba.jit(nopython=True, cache=True)
def _blend_multiply(block1, block2):
    # Dask blocks might be float, so we work in float and clip
    result = (block1.astype(np.float32) * block2.astype(np.float32)) / 255.0
    return np.clip(result, 0, 255)

@numba.jit(nopython=True, cache=True)
def _blend_screen(block1, block2):
    result = 1.0 - (1.0 - block1.astype(np.float32)/255.0) * (1.0 - block2.astype(np.float32)/255.0)
    return np.clip(result * 255.0, 0, 255)

@numba.jit(nopython=True, cache=True)
def _blend_overlay(block1, block2):
    # Using a loop for clarity, Numba will optimize it
    result = np.zeros_like(block1, dtype=np.float32)
    b1_float = block1.astype(np.float32) / 255.0
    b2_float = block2.astype(np.float32) / 255.0
    for i in range(b1_float.shape[0]):
        for j in range(b1_float.shape[1]):
            for k in range(b1_float.shape[2]):
                base = b1_float[i,j,k]
                blend = b2_float[i,j,k]
                if base < 0.5:
                    result[i,j,k] = 2.0 * base * blend
                else:
                    result[i,j,k] = 1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    return np.clip(result * 255.0, 0, 255)

def apply_blend_mode(dask_array, op):
    """
    Applies a pixel-wise blend mode.
    For this to work, we assume another image/stack is provided or generated.
    This is a placeholder for how it *would* work. In a real scenario, you'd
    load or generate the second image stack. For this demo, we blend the
    image with itself.
    """
    blend_layer = dask_array

    mode_map = {
        "multiply": _blend_multiply,
        "screen": _blend_screen,
        "overlay": _blend_overlay
    }

    blend_func = mode_map.get(op.blend_mode)

    if blend_func:
        return da.map_blocks(blend_func, dask_array, blend_layer, dtype=np.uint8)
    else:
        logging.warning(f"Unknown blend mode: {op.blend_mode}. Returning original array.")
        return dask_array

def apply_lut(dask_array, op):
    """
    Generates or loads a LUT based on the operation's parameters and applies it.
    """
    lut_params = op.lut_params
    lut_table = None

    try:
        if lut_params.lut_source == "file":
            lut_table = lut_manager.load_lut(lut_params.fixed_lut_path)
        else: # generated
            gen_func_name = f"generate_{lut_params.lut_generation_type}_lut"
            gen_func = getattr(lut_manager, gen_func_name, lut_manager.generate_linear_lut)

            # This is a simplified call; a real version would pass more params
            if lut_params.lut_generation_type == "gamma":
                lut_table = gen_func(lut_params.gamma_value, 0, 255, 0, 255)
            else:
                lut_table = gen_func(0, 255, 0, 255)

    except Exception as e:
        logging.error(f"Failed to generate or load LUT: {e}")

    if lut_table is None:
        logging.warning("Using default pass-through LUT.")
        lut_table = lut_manager.get_default_z_lut()

    return dask_array.map_blocks(
        lambda block: lut_table[block],
        dtype=np.uint8
    )

def _read_and_prepare_image(path):
    """Reads an image and converts it to 8-bit grayscale if necessary."""
    try:
        img = imageio.imread(path)
        if img.ndim == 3:
            # Convert RGB to grayscale
            img = img_as_ubyte(rgb2gray(img))
        elif img.dtype != np.uint8:
            # Ensure image is 8-bit
            img = img.astype(np.uint8)
        return img
    except Exception as e:
        logging.error(f"Could not read or process image {path}: {e}")
        return None


def create_dask_stack(image_folder: str) -> da.Array:
    """
    Creates a 3D Dask array from a folder of images, representing a Z-stack.

    Args:
        image_folder: The path to the folder containing numbered PNG images.

    Returns:
        A 3D Dask array with dimensions (Z, Y, X), chunked in (64, 64, 64).

    Raises:
        FileNotFoundError: If the image_folder does not exist.
        ValueError: If no valid images are found in the folder.
    """
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"Input directory not found: {image_folder}")

    # Robustly find and sort image files based on numbers in the filename
    numeric_pattern = re.compile(r'(\d+)')
    def get_numeric_part(filename):
        parts = numeric_pattern.findall(filename)
        return int(''.join(parts)) if parts else float('inf')

    try:
        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.lower().endswith('.png')],
            key=get_numeric_part
        )
    except Exception as e:
        logging.error(f"Could not sort files in {image_folder}: {e}")
        image_files = []

    if not image_files:
        raise ValueError(f"No PNG images found in '{image_folder}'")

    image_paths = [os.path.join(image_folder, f) for f in image_files]

    # Read the first image to determine shape and dtype for the stack
    first_image = _read_and_prepare_image(image_paths[0])
    if first_image is None:
        raise ValueError(f"Could not read the first image: {image_paths[0]}")

    img_shape = first_image.shape
    img_dtype = first_image.dtype
    logging.info(f"Detected image properties: Shape={img_shape}, Dtype={img_dtype}")

    # Create a list of delayed read operations
    lazy_imread = dask.delayed(_read_and_prepare_image)
    lazy_images = [lazy_imread(path) for path in image_paths]

    # Create the Dask array from the stack of delayed images
    # The final array shape will be (num_images, height, width)
    stack_shape = (len(lazy_images),) + img_shape

    dask_stack = da.from_delayed(
        lazy_images, shape=stack_shape, dtype=img_dtype
    )

    # Rechunk the array to the desired 3D chunk size
    # Dask handles the rechunking operation efficiently.
    chunk_size = (64, 64, 64)
    dask_stack = dask_stack.rechunk(chunk_size)

    logging.info(f"Created Dask array with shape {dask_stack.shape} and chunks {dask_stack.chunksize}")

    return dask_stack
