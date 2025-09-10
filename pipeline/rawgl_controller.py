import subprocess
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import concurrent.futures

@dataclass
class RawGLJob:
    """A dataclass to hold the parameters for a single RawGL processing job."""
    shader_path: Path
    input_path: Path
    output_path: Path
    pass_size: tuple
    uniforms: Dict[str, Any] = None

class RawGLController:
    """
    A controller to manage and execute processing pipelines using the
    RawGL command-line tool.
    """
    def __init__(self, executable_path: str):
        """
        Initializes the controller with the path to the RawGL executable.

        Args:
            executable_path (str): The full path to the rawgl.exe executable.
        """
        self.executable_path = Path(executable_path)
        if not self.executable_path.is_file():
            raise FileNotFoundError(
                f"RawGL executable not found at: {self.executable_path}"
            )

    def build_cli_args(self,
                         shader_path: Path,
                         input_path: Path,
                         output_path: Path,
                         pass_size: tuple,
                         uniforms: Dict[str, Any] = None) -> List[str]:
        """
        Builds the list of command-line arguments for a single RawGL process.

        This method abstracts the CLI switches for processing a single 8-bit
        grayscale PNG image.

        Args:
            shader_path (Path): Path to the GLSL fragment shader.
            input_path (Path): Path to the input PNG image.
            output_path (Path): Path for the output PNG image.
            pass_size (tuple): A tuple (width, height) for the processing resolution.
            uniforms (Dict[str, Any]): A dictionary of additional uniforms to pass
                                       to the shader.

        Returns:
            List[str]: A list of strings representing the command-line arguments.
        """
        if not shader_path.is_file():
            raise FileNotFoundError(f"Shader file not found: {shader_path}")
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        args = [
            str(self.executable_path),
            # Define the shader pass. We assume a simple pass-through vertex shader
            # is either built-in or not needed for a simple fragment shader pass.
            # RawGL can often work with just a fragment shader for 2D processing.
            "-P", str(shader_path),
            # Define the output size of the pass
            "--pass_size", str(pass_size[0]), str(pass_size[1]),
            # Define the input texture
            "--in", "in_texture", str(input_path),
            # Define the output file and its format for 8-bit grayscale
            "--out", "out_color", str(output_path),
            "--out_format", "r8",      # 1 channel, 8-bit unsigned normalized
            "--out_channels", "1",
            "--out_bits", "8",
        ]

        # Add any additional uniforms
        if uniforms:
            for name, value in uniforms.items():
                args.extend(["--in", name, str(value)])

        return args

    def run_single_job(self, args: List[str]):
        """
        Executes a single RawGL job using subprocess.

        Args:
            args (List[str]): The list of command-line arguments.

        Returns:
            A tuple (return_code, stdout, stderr).
        """
        try:
            process = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=False  # Don't raise exception on non-zero exit codes
            )
            return process.returncode, process.stdout, process.stderr
        except FileNotFoundError:
            raise RuntimeError(f"Failed to execute RawGL. Is the path correct? {args[0]}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while running RawGL: {e}")

    def _execute_job(self, job: RawGLJob) -> Dict[str, Any]:
        """Helper method to execute a single job and return its result."""
        try:
            args = self.build_cli_args(
                shader_path=job.shader_path,
                input_path=job.input_path,
                output_path=job.output_path,
                pass_size=job.pass_size,
                uniforms=job.uniforms
            )

            # This print is helpful for debugging but can be noisy.
            # print(f"Executing job for input: {job.input_path}")
            returncode, stdout, stderr = self.run_single_job(args)

            if returncode == 0:
                return {"status": "success", "output_path": job.output_path, "stdout": stdout}
            else:
                return {"status": "failure", "input_path": job.input_path, "stderr": stderr}

        except Exception as e:
            return {"status": "error", "input_path": job.input_path, "error": str(e)}

    def run_pipeline(self, jobs: List[RawGLJob], max_workers: int = 4):
        """
        Runs a pipeline of RawGL jobs in parallel using a thread pool.

        Args:
            jobs (List[RawGLJob]): A list of jobs to execute.
            max_workers (int): The maximum number of parallel processes to run.
        """
        print(f"\n--- Starting RawGL pipeline with {max_workers} workers for {len(jobs)} jobs ---")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs to the executor
            future_to_job = {executor.submit(self._execute_job, job): job for job in jobs}

            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        print(f"  [SUCCESS] Job for {job.input_path} finished.")
                    else:
                        print(f"  [FAILURE] Job for {job.input_path} failed: {result.get('stderr') or result.get('error')}")
                except Exception as exc:
                    print(f"  [ERROR] Job for {job.input_path} generated an exception: {exc}")
        print("--- RawGL pipeline finished ---")


if __name__ == '__main__':
    print("--- RawGLController Test ---")

    dummy_exe = "rawgl.exe"
    dummy_shader = Path("test.frag")

    # Setup a temporary directory for test files
    test_dir = Path("./temp_rawgl_test")
    test_dir.mkdir(exist_ok=True)

    (test_dir / dummy_shader).touch()
    Path(dummy_exe).touch()

    controller = RawGLController(dummy_exe)
    print(f"Controller initialized with dummy executable: {controller.executable_path}")

    # Create a list of dummy jobs
    jobs = []
    num_jobs = 5
    for i in range(num_jobs):
        input_path = test_dir / f"input_{i}.png"
        output_path = test_dir / f"output_{i}.png"
        input_path.touch() # Create dummy input file

        jobs.append(RawGLJob(
            shader_path=test_dir / dummy_shader,
            input_path=input_path,
            output_path=output_path,
            pass_size=(128, 128)
        ))

    print(f"\nCreated {len(jobs)} dummy jobs for the pipeline.")

    # --- Test run_pipeline ---
    # We can't actually run rawgl.exe, but we can test the pipeline machinery.
    # To do this, we'll mock the run_single_job method.
    def mock_run_single_job(self, args: List[str]):
        # Find the output file path in the args to simulate its creation
        output_arg_index = args.index('--out_color') + 1
        output_path = Path(args[output_arg_index])
        output_path.touch() # Simulate file creation
        return (0, "mock stdout", "") # Simulate success

    # Temporarily replace the real method with the mock
    RawGLController.run_single_job = mock_run_single_job

    controller.run_pipeline(jobs, max_workers=2)

    # Verify that mock output files were created
    print("\n--- Verifying pipeline outputs ---")
    success = True
    for i in range(num_jobs):
        expected_output = test_dir / f"output_{i}.png"
        if expected_output.exists():
            print(f"  [OK] Output file found: {expected_output}")
        else:
            print(f"  [FAIL] Output file missing: {expected_output}")
            success = False

    if success:
        print("Pipeline execution test PASSED.")
    else:
        print("Pipeline execution test FAILED.")

    # --- Clean up ---
    print("\n--- Cleaning up test files ---")
    Path(dummy_exe).unlink()
    for f in test_dir.iterdir():
        f.unlink()
    test_dir.rmdir()
    print("Cleanup complete.")
