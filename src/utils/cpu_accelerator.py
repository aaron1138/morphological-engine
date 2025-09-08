import numpy as np
import numba

# Note: In a real-world scenario, this kind of element-wise operation would be
# done with NumPy's vectorized functions (e.g., np.sin(array) * ...), which
# is already highly optimized in C. This example uses explicit loops to
# simulate a more complex, custom algorithm where vectorization is not possible,
# which is where Numba truly shines.

def process_array_python(array: np.ndarray) -> np.ndarray:
    """
    A sample CPU-bound function in pure Python that processes a NumPy array.
    This function uses explicit loops to simulate a complex algorithm that
    cannot be easily vectorized in NumPy.

    Args:
        array (np.ndarray): A 2D NumPy array.

    Returns:
        A new 2D NumPy array with the processed values.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")

    height, width = array.shape
    output_array = np.empty_like(array)

    for y in range(height):
        for x in range(width):
            # A sample calculation
            val = array[y, x]
            output_array[y, x] = (val * 0.5) + (np.sin(val)**2) - (np.cos(val)**2)

    return output_array

@numba.jit(nopython=True, cache=True)
def process_array_numba(array: np.ndarray) -> np.ndarray:
    """
    An identical function to process_array_python, but accelerated with
    Numba's JIT compiler. The @numba.jit decorator compiles this function
    to fast machine code.

    Args:
        array (np.ndarray): A 2D NumPy array.

    Returns:
        A new 2D NumPy array with the processed values.
    """
    if array.ndim != 2:
        # Numba can't raise ValueError with a string, so we use a bare raise
        # for type-stability. Or handle error checking outside.
        # For this example, we assume valid input.
        pass

    height, width = array.shape
    output_array = np.empty_like(array)

    for y in range(height):
        for x in range(width):
            # The exact same calculation as the pure Python version
            val = array[y, x]
            output_array[y, x] = (val * 0.5) + (np.sin(val)**2) - (np.cos(val)**2)

    return output_array
