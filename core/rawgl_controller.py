import subprocess
import shlex

class RawGLController:
    """
    Manages the construction and execution of RawGL command-line processes.
    """

    def __init__(self, rawgl_executable_path):
        self.executable_path = rawgl_executable_path

    def build_command(self, config):
        """
        Builds a command-line argument list from a configuration dictionary.

        Args:
            config (dict): A dictionary containing the parameters for a single RawGL pass.
                           Example:
                           {
                               "pass_vertfrag": "path/to/shader.glsl",
                               "pass_size": "1024 1024",
                               "in": "Texture0 path/to/input.png",
                               "out": "OutColor path/to/output.png",
                               "out_channels": 1,
                               "out_bits": 8,
                               "out_format": "r8"
                           }

        Returns:
            list: A list of strings representing the command and its arguments.
        """
        command = [self.executable_path]
        for key, value in config.items():
            command.append(f"--{key}")
            # shlex.split handles cases where a value has spaces, like "Texture0 path/to/input.png"
            if isinstance(value, str) and ' ' in value:
                 command.extend(shlex.split(value))
            else:
                command.append(str(value))
        return command

    def run_pass(self, config):
        """
        Runs a single RawGL pass with the given configuration.

        Args:
            config (dict): The configuration for the pass.

        Returns:
            A tuple (success, output_log).
            success (bool): True if the process completed with a zero exit code, False otherwise.
            output_log (str): The combined stdout and stderr from the process.
        """
        command = self.build_command(config)
        print(f"Executing RawGL command: {' '.join(command)}")

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False  # Don't raise exception on non-zero exit code
            )

            log = f"--- stdout ---\n{process.stdout}\n"
            log += f"--- stderr ---\n{process.stderr}"

            if process.returncode != 0:
                print(f"RawGL process failed with exit code {process.returncode}")
                return False, log

            print("RawGL process completed successfully.")
            return True, log

        except FileNotFoundError:
            error_msg = f"Error: The RawGL executable was not found at '{self.executable_path}'."
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred while running RawGL: {e}"
            print(error_msg)
            return False, error_msg
