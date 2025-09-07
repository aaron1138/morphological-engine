"""
Utility module for performance-critical functions.

This module uses Numba for Just-In-Time (JIT) compilation to accelerate
CPU-bound mathematical operations.
"""

try:
    from numba import jit, njit
except ImportError:
    print("WARNING: numba is not installed. JIT compilation will not be available.")
    # Create a dummy decorator if numba is not present
    def njit(func):
        return func
    def jit(func):
        return func

import numpy as np

@njit
def sample_jit_function(x, y):
    """
    A sample function accelerated by Numba's JIT compiler.

    This is a placeholder to demonstrate the use of @njit.
    It performs a simple element-wise operation on two numpy arrays.

    Args:
        x (np.ndarray): A numpy array.
        y (np.ndarray): Another numpy array of the same shape.

    Returns:
        A new numpy array containing the result.
    """
    # This loop will be compiled to fast machine code by Numba
    return np.tanh(x) + np.sqrt(y)
