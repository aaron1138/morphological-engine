import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def sample_grid_slice_numba(accessor_buffer, min_val, max_val, slice_z, width, height, center_x, center_y):
    """
    Samples a 2D slice from an OpenVDB grid accessor at a specific Z depth.
    This function is JIT-compiled with Numba for performance.

    NOTE: Numba cannot work directly with the pyopenvdb Accessor object.
    This function is a placeholder for the *pattern* of using Numba. A real
    implementation would require passing Numba-compatible data, such as a
    raw pointer or a NumPy representation of the tree if possible.

    For this example, we simulate the sampling with a simple mathematical function
    to demonstrate the performance gain of Numba on loop-heavy code.

    Args:
        accessor_buffer: A placeholder for data that Numba can understand.
        min_val (float): The minimum value from the grid, for normalization.
        max_val (float): The maximum value from the grid, for normalization.
        slice_z (int): The integer Z-coordinate for the slice.
        width (int): The width of the slice to sample.
        height (int): The height of the slice to sample.
        center_x (int): The X-coordinate of the center of the slice.
        center_y (int): The Y-coordinate of the center of the slice.

    Returns:
        A 2D NumPy array containing the sampled voxel values, normalized to [0, 1].
    """
    slice_data = np.zeros((height, width), dtype=np.float32)
    value_range = max(abs(min_val), abs(max_val))
    if value_range == 0:
        value_range = 1.0

    start_x = center_x - width // 2
    start_y = center_y - height // 2

    for y in range(height):
        for x in range(width):
            # In a real scenario, we would read from the accessor_buffer.
            # Here, we simulate a value based on distance from center,
            # which is a pattern similar to a sphere level set.
            dist_x = (start_x + x) - center_x
            dist_y = (start_y + y) - center_y
            dist_z = slice_z
            simulated_value = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2) - (width / 3.0)

            # Normalize the signed distance field value to a 0-1 range.
            normalized_value = (simulated_value / value_range + 1.0) / 2.0
            slice_data[y, x] = min(max(normalized_value, 0.0), 1.0)

    return slice_data

def sample_grid_slice_python(grid, slice_z, width, height, center_x, center_y):
    """
    Samples a 2D slice from an OpenVDB grid at a specific Z depth.
    This is the pure Python version for comparison.
    """
    accessor = grid.getConstAccessor()
    slice_data = np.zeros((height, width), dtype=np.float32)

    min_val, max_val = grid.minMaxValues()
    value_range = max(abs(min_val), abs(max_val))
    if value_range == 0:
        value_range = 1.0

    start_x = center_x - width // 2
    start_y = center_y - height // 2

    for y in range(height):
        for x in range(width):
            coord = (start_x + x, start_y + y, slice_z)
            value = accessor.getValue(coord)

            normalized_value = (value / value_range + 1.0) / 2.0
            slice_data[y, x] = min(max(normalized_value, 0.0), 1.0)

    return slice_data
