import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os
from pathlib import Path

# Make the src directory available for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.rawgl_controller import RawGLWorker

class TestRawGLController(unittest.TestCase):

    @patch('src.processing.rawgl_controller.subprocess.run')
    def test_command_generation_simple(self, mock_subprocess_run):
        """Tests that a simple, single-step pipeline generates the correct command."""
        mock_subprocess_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        pipeline = [
            {
                'pass_comp': 'shader.comp',
                'in': {'Texture0': 'input.png'},
                'out': {'OutColor': 'output.png'},
                'out_format': 'r8',
                'out_channels': 1,
                'out_bits': 8
            }
        ]

        worker = RawGLWorker(pipeline, rawgl_executable="rawgl_test")
        worker.run()

        expected_command = [
            "rawgl_test",
            "--pass_comp", "shader.comp",
            "--in", "Texture0", "input.png",
            "--out", "OutColor", "output.png",
            "--out_format", "r8",
            "--out_channels", "1",
            "--out_bits", "8"
        ]

        mock_subprocess_run.assert_called_once_with(
            expected_command,
            capture_output=True, text=True, check=True
        )

    @patch('src.processing.rawgl_controller.subprocess.run')
    def test_temp_file_chaining(self, mock_subprocess_run):
        """Tests that intermediate temp files are correctly chained between passes."""
        mock_subprocess_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        pipeline = [
            { # Step 0: Input -> Temp
                'pass_comp': 'step1.comp',
                'in': {'Texture0': 'input.png'},
                'out': {'OutColor': 'TEMP'}
            },
            { # Step 1: Temp -> Output
                'pass_comp': 'step2.comp',
                'in': {'Texture0': (0, 'OutColor')}, # Reference output from step 0
                'out': {'FinalImage': 'final.png'}
            }
        ]

        worker = RawGLWorker(pipeline)
        worker.run()

        self.assertEqual(mock_subprocess_run.call_count, 2)

        # Check the first call (writes to a temp file)
        first_call_args = mock_subprocess_run.call_args_list[0].args[0]
        self.assertEqual(first_call_args[0:8], ['rawgl', '--pass_comp', 'step1.comp', '--in', 'Texture0', 'input.png', '--out', 'OutColor'])
        temp_output_path = Path(first_call_args[8]) # The path of the temp output file
        self.assertTrue(temp_output_path.name.startswith("step_0_OutColor"))
        self.assertTrue(temp_output_path.name.endswith(".png"))

        # Check the second call (reads from the temp file)
        second_call_args = mock_subprocess_run.call_args_list[1].args[0]
        # Expected: rawgl --pass_comp step2.comp --in Texture0 /path/to/temp --out FinalImage final.png
        self.assertEqual(second_call_args[6], '--out')
        self.assertEqual(second_call_args[7], 'FinalImage')
        self.assertEqual(second_call_args[8], 'final.png')

        input_to_second_step = second_call_args[5]
        self.assertEqual(input_to_second_step, str(temp_output_path))

if __name__ == '__main__':
    unittest.main()
