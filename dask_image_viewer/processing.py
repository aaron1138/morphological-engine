import numba
import numpy as np

@numba.jit(nopython=True, cache=True)
def example_numba_filter(image_chunk):
    """
    An example of a Numba-accelerated filter that could be applied to a chunk of the Dask array.
    This is just a placeholder to demonstrate Numba integration.
    """
    # Create a new array to store the output
    output_array = np.zeros_like(image_chunk)

    # Numba will heavily optimize this loop
    for y in range(image_chunk.shape[0]):
        for x in range(image_chunk.shape[1]):
            # Example operation
            output_array[y, x] = image_chunk[y, x] * 2

    return output_array
