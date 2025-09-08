import pytest
import numpy as np
import time
import os

# Make the src directory available for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.cpu_accelerator import process_array_python, process_array_numba

@pytest.fixture(scope="module")
def sample_array():
    """
    Provides a sample array for the performance test.
    Scope is 'module' so it's only created once.
    """
    return np.random.rand(512, 512).astype(np.float64)

def test_numba_correctness(sample_array):
    """
    Tests that the Numba-jitted function produces the same result as
    the pure Python version.
    """
    print("\nTesting Numba function for correctness...")
    python_result = process_array_python(sample_array)
    numba_result = process_array_numba(sample_array)

    assert np.allclose(python_result, numba_result), "Numba result does not match Python result."
    print("Correctness test passed.")

def test_numba_performance(sample_array, capsys):
    """
    A simple performance comparison to demonstrate Numba's speedup.
    This is not a rigorous benchmark but serves as a demonstration.
    """
    # --- Time the pure Python version ---
    start_time_py = time.perf_counter()
    process_array_python(sample_array)
    end_time_py = time.perf_counter()
    python_duration = end_time_py - start_time_py

    # --- Time the Numba version ---
    # First run is for JIT compilation, so we don't time it.
    process_array_numba(sample_array)

    start_time_nb = time.perf_counter()
    process_array_numba(sample_array)
    end_time_nb = time.perf_counter()
    numba_duration = end_time_nb - start_time_nb

    # Use capsys to capture print output for better test reporting
    with capsys.disabled():
        print(f"\n--- Numba Performance Demonstration ---")
        print(f"Pure Python execution time: {python_duration:.6f} seconds")
        print(f"Numba execution time (after compile): {numba_duration:.6f} seconds")

        # Avoid division by zero if numba is somehow slower or timing is weird
        if numba_duration > 0:
            speedup = python_duration / numba_duration
            print(f"Speedup: {speedup:.2f}x")
            # Assert that the Numba version is at least 5x faster.
            # This is a reasonable expectation for this type of calculation.
            assert speedup > 5
        else:
            # If numba_duration is zero, it's clearly faster.
            assert True

        print("---------------------------------------")
