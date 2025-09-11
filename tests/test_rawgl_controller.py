import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Make the src directory available for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.rawgl_controller import RawGLDaskWorker

class MockDaskGrid:
    """A mock DaskGrid for testing the controller."""
    def __init__(self, shape=(10, 100, 100)):
        self.shape = shape

    def get_slice(self, plane, index):
        # This doesn't need to return a real Dask array for this test
        return f"slice_{plane}_{index}"

class TestRawGLDaskController(unittest.TestCase):

    @patch('src.processing.rawgl_controller.dask.compute', return_value=([],))
    @patch('src.processing.rawgl_controller.process_slice_with_rawgl')
    @patch('src.processing.rawgl_controller.Client', MagicMock())
    @patch('src.processing.rawgl_controller.LocalCluster', MagicMock())
    def test_dask_worker_orchestration(self, mock_process_function, mock_compute):
        """
        Tests that the worker calls the processing function with the correct arguments
        for each slice in the specified range.
        """
        # --- Test Data ---
        mock_grid = MockDaskGrid(shape=(10, 100, 100))
        config = {
            'num_workers': 2,
            'plane': 'YZ',
            'slice_range': (2, 5), # Process slices 2, 3, 4 (3 total)
            'shader_pass': {'pass_comp': 'test.comp'},
            'rawgl_executable': 'rawgl_test'
        }

        # --- Execution ---
        # The dask.delayed wrapper is transparent here since we mock the target function
        # We use a side_effect to replace the dask.delayed decorator with a function
        # that simply returns the function it's wrapping. This means when the code
        # calls the 'delayed' function, it's actually calling our mock of
        # process_slice_with_rawgl directly.
        with patch('src.processing.rawgl_controller.dask.delayed', side_effect=lambda x: x):
             worker = RawGLDaskWorker(mock_grid, config)
             worker.run()

        # --- Assertions ---
        # 1. Was the processing function called for each slice in the range?
        self.assertEqual(mock_process_function.call_count, 3)

        # 2. Check the arguments of the first call
        first_call_args = mock_process_function.call_args_list[0].args
        self.assertEqual(first_call_args[0], mock_grid)             # dask_grid
        self.assertEqual(first_call_args[1], 'YZ')                  # plane
        self.assertEqual(first_call_args[2], 2)                      # slice_index
        self.assertEqual(first_call_args[3], config['shader_pass'])  # shader_pass
        self.assertEqual(first_call_args[4], 'rawgl_test')           # rawgl_executable
        self.assertIsInstance(first_call_args[5], Path)              # output_dir (Path object)

        # 3. Check the slice index for the last call
        last_call_args = mock_process_function.call_args_list[2].args
        self.assertEqual(last_call_args[2], 4) # slice_range is exclusive (2, 3, 4)

        # 4. Check that dask.compute was called.
        # The *tasks syntax means we can't easily inspect the list, but we can ensure it was called.
        mock_compute.assert_called_once()

if __name__ == '__main__':
    unittest.main()
