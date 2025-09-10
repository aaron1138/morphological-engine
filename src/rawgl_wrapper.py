import subprocess
import shlex
from typing import List, Tuple, Dict, Optional

class RawGLWrapper:
    """
    A Python wrapper for the RawGL command-line interface.

    This class helps construct and execute RawGL commands for applying shaders
    to images. It includes specific helpers for common formats, like 8-bit
    grayscale PNGs.
    """

    def __init__(self, rawgl_executable: str):
        """
        Initializes the wrapper with the path to the RawGL executable.

        Args:
            rawgl_executable (str): The file path to the rawgl binary.
        """
        self.executable = rawgl_executable
        self.commands = []

    def add_pass(self,
                 shader_path: str,
                 output_size: Tuple[int, int],
                 output_path: str,
                 inputs: Dict[str, str],
                 is_greyscale_png: bool = False):
        """
        Adds a new rendering pass to the command sequence.

        Args:
            shader_path (str): Path to the fragment shader file (.frag).
            output_size (Tuple[int, int]): The (width, height) of the output.
            output_path (str): The path to save the output image.
            inputs (Dict[str, str]): A dictionary mapping uniform names to input texture paths.
                                     e.g., {'Texture0': 'input.png'}
            is_greyscale_png (bool): If True, configures the output for 8-bit single-channel PNG.
        """
        pass_cmd = []

        # Shader pass
        # Assuming a single fragment shader for now, as per the user's focus on image processing.
        # RawGL seems to handle this with --pass_vertfrag and a default v-shader if only one file is given.
        # Or, if it's a .frag, it might need a generic vertex shader. We'll assume a .frag for simplicity.
        # For robustness, we'll use a placeholder for a potential vertex shader.
        pass_cmd.extend(['--pass_vertfrag', 'passthrough.vert', shader_path])

        # Output size
        pass_cmd.extend(['--pass_size', str(output_size[0]), str(output_size[1])])

        # Input textures
        for uniform_name, texture_path in inputs.items():
            pass_cmd.extend(['--in', uniform_name, texture_path])

        # Output file
        pass_cmd.extend(['--out', 'OutColor', output_path])

        # Format helpers
        if is_greyscale_png:
            if not output_path.lower().endswith('.png'):
                print("Warning: Greyscale output is specified, but output path is not a .png file.")
            # r8 = 1 channel, 8 bits. Perfect for grayscale.
            pass_cmd.extend(['--out_format', 'r8'])
            pass_cmd.extend(['--out_channels', '1'])
            pass_cmd.extend(['--out_bits', '8'])

        self.commands.append(pass_cmd)

    def build_command(self) -> List[str]:
        """Constructs the full command-line argument list."""
        full_cmd = [self.executable]
        for pass_cmd in self.commands:
            full_cmd.extend(pass_cmd)
        return full_cmd

    def run(self) -> Tuple[bool, str]:
        """
        Executes the configured RawGL command.

        Returns:
            A tuple containing:
            - bool: True if the command executed successfully (return code 0), False otherwise.
            - str: The combined stdout and stderr from the command.
        """
        command_list = self.build_command()
        command_str = shlex.join(command_list)
        print(f"Executing RawGL command:\n{command_str}")

        try:
            process = subprocess.run(
                command_list,
                capture_output=True,
                text=True,
                check=False  # We check the returncode manually
            )

            output = process.stdout + process.stderr
            if process.returncode == 0:
                print("RawGL executed successfully.")
                return True, output
            else:
                print(f"RawGL execution failed with return code {process.returncode}.")
                return False, output

        except FileNotFoundError:
            error_msg = f"Error: The executable '{self.executable}' was not found."
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred while running RawGL: {e}"
            print(error_msg)
            return False, error_msg

if __name__ == '__main__':
    # This example demonstrates how to use the RawGLWrapper.
    # It requires the RawGL executable to be available at the specified path.

    # On Windows, this might be 'C:/path/to/rawgl.exe'
    # On Linux/macOS, this might be '/usr/local/bin/rawgl'
    # We use a placeholder here.
    RAWGL_PATH = "path/to/your/rawgl"

    print("--- Running RawGLWrapper Example ---")
    wrapper = RawGLWrapper(RAWGL_PATH)

    # Define a processing job
    wrapper.add_pass(
        shader_path="invert_color.frag",
        output_size=(512, 512),
        output_path="output_inverted.png",
        inputs={'u_texture': 'input.png'},
        is_greyscale_png=True
    )

    # The wrapper is now configured. The run() method would execute it.
    # We will just print the command here since we don't have the executable.

    final_command = wrapper.build_command()
    print("\n--- Generated Command ---")
    print(shlex.join(final_command))
    print("\n--- Example Finished ---")
    # In a real scenario, you would call:
    # success, output = wrapper.run()
    # print(output)
