import dask
import dask.array as da
import numpy as np
from PIL import Image
import tempfile
import os
from typing import Tuple

from .rawgl_wrapper import RawGLWrapper

def process_slice_with_rawgl(slice_2d: np.ndarray, rawgl_config: dict) -> np.ndarray:
    """
    Processes a single 2D NumPy array (a slice) using RawGL.

    This function is intended to be mapped over a Dask array.

    Args:
        slice_2d (np.ndarray): The 2D array representing the image slice.
        rawgl_config (dict): A dictionary with configuration for RawGL, including
                             'executable', 'shader_path'.

    Returns:
        np.ndarray: The processed 2D array.
    """
    try:
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_in:
            input_path = temp_in.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_out:
            output_path = temp_out.name

        # Save the input slice to the temporary file
        Image.fromarray(slice_2d).save(input_path)

        # Configure and run RawGL
        wrapper = RawGLWrapper(rawgl_config['executable'])
        wrapper.add_pass(
            shader_path=rawgl_config['shader_path'],
            output_size=slice_2d.shape[::-1],  # (width, height)
            output_path=output_path,
            inputs={'u_texture': input_path},
            is_greyscale_png=True
        )

        success, output = wrapper.run()
        if not success:
            print(f"RawGL processing failed for a slice. Error: {output}")
            # Return the original slice on failure
            return slice_2d

        # Load the processed image back
        with Image.open(output_path) as img:
            processed_array = np.array(img)

        return processed_array

    finally:
        # Clean up temporary files
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)

def run_orthogonal_pipeline(dask_volume: da.Array, rawgl_config: dict, num_workers: int):
    """
    Runs a parallel processing pipeline on a Dask array, extracting and
    processing orthogonal XZ and YZ slices.

    Args:
        dask_volume (da.Array): The input 3D Dask array (Z, Y, X).
        rawgl_config (dict): Configuration for the RawGL wrapper.
        num_workers (int): The number of parallel workers for Dask to use.

    Returns:
        A tuple containing the two processed Dask arrays: (processed_yz, processed_xz)
    """
    if dask_volume.ndim != 3:
        raise ValueError("Input must be a 3D Dask array.")

    print("\n--- Starting Orthogonal Dask Pipeline ---")
    print(f"Using {num_workers} Dask workers.")

    # --- Process YZ slices ---
    # Transpose from (Z, Y, X) to (X, Z, Y) to iterate over X
    yz_view = dask_volume.transpose((2, 0, 1))
    print(f"Processing YZ slices. View shape: {yz_view.shape}")

    # Map the processing function over the YZ slices
    processed_yz_lazy = yz_view.map_blocks(
        process_slice_with_rawgl,
        rawgl_config=rawgl_config,
        dtype=dask_volume.dtype
    )

    # --- Process XZ slices ---
    # Transpose from (Z, Y, X) to (Y, Z, X) to iterate over Y
    xz_view = dask_volume.transpose((1, 0, 2))
    print(f"Processing XZ slices. View shape: {xz_view.shape}")

    # Map the processing function over the XZ slices
    processed_xz_lazy = xz_view.map_blocks(
        process_slice_with_rawgl,
        rawgl_config=rawgl_config,
        dtype=dask_volume.dtype
    )

    # --- Compute the results in parallel ---
    print("\nTriggering Dask computation for both pipelines...")
    with dask.config.set(scheduler='threads', num_workers=num_workers):
        (processed_yz, processed_xz) = dask.compute(processed_yz_lazy, processed_xz_lazy)

    print("Dask computation finished.")

    # The results are NumPy arrays. We can wrap them back in Dask arrays if needed.
    processed_yz_da = da.from_array(processed_yz, chunks=processed_yz_lazy.chunksize)
    processed_xz_da = da.from_array(processed_xz, chunks=processed_xz_lazy.chunksize)

    return processed_yz_da, processed_xz_da


if __name__ == '__main__':
    # This example demonstrates how to use the Dask pipeline.
    # It requires the image_loader and a dummy image stack.
    from .image_loader import load_png_stack_as_dask_array

    print("--- Running Dask Pipeline Example ---")

    dummy_dir = "temp_png_stack_pipeline"
    if not os.path.exists(dummy_dir):
        os.makedirs(dummy_dir)

    print(f"Creating dummy PNG files in '{dummy_dir}'...")
    for i in range(5): # Using fewer slices for a quick example
        img_data = np.zeros((64, 64), dtype=np.uint8)
        img_data[10:54, 10 + i * 5:20 + i * 5] = 255 # A moving bar
        img = Image.fromarray(img_data, 'L')
        img.save(os.path.join(dummy_dir, f"slice_{i:03d}.png"))

    # Load the volume
    dask_volume = load_png_stack_as_dask_array(dummy_dir)

    if dask_volume is not None:
        # Define mock RawGL config
        mock_rawgl_config = {
            'executable': 'path/to/rawgl', # Placeholder
            'shader_path': 'shaders/invert.frag'
        }

        # Run the pipeline
        yz_result, xz_result = run_orthogonal_pipeline(
            dask_volume,
            mock_rawgl_config,
            num_workers=2
        )

        print("\n--- Pipeline Results ---")
        print(f"Processed YZ volume shape: {yz_result.shape}")
        print(f"Processed XZ volume shape: {xz_result.shape}")

        # Save a sample output slice to verify
        sample_slice = yz_result[0].compute()
        Image.fromarray(sample_slice).save("processed_yz_sample.png")
        print("\nSaved a sample processed slice to 'processed_yz_sample.png'")

    # Clean up
    print("\nCleaning up dummy files...")
    for f in os.listdir(dummy_dir):
        os.remove(os.path.join(dummy_dir, f))
    os.rmdir(dummy_dir)

    print("\n--- Dask Pipeline Example Finished ---")
