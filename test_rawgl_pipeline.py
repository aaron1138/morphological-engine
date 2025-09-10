import sys
from pathlib import Path
import numpy as np
from typing import List

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.rawgl_controller import RawGLController, RawGLJob
from data_io.image import write_image

def setup_test_environment(test_dir: Path):
    """Creates all necessary dummy files and directories for the test."""
    print(f"--- Setting up test environment in {test_dir} ---")
    test_dir.mkdir(exist_ok=True)

    # 1. Create dummy rawgl.exe
    (test_dir / "rawgl.exe").touch()

    # 2. Create dummy invert shader
    shader_path = test_dir / "invert.frag"
    invert_shader_code = """
        #version 330
        uniform sampler2D in_texture;
        in vec2 v_text;
        out vec4 out_color;

        void main() {
            vec4 tex_color = texture(in_texture, v_text);
            // Invert the grayscale value (stored in the 'r' component)
            out_color = vec4(1.0 - tex_color.r, 1.0 - tex_color.r, 1.0 - tex_color.r, 1.0);
        }
    """
    shader_path.write_text(invert_shader_code)

    # 3. Create dummy input images
    num_images = 4
    image_shape = (32, 32)
    for i in range(num_images):
        input_path = test_dir / f"input_{i}.png"
        data = np.full(image_shape, fill_value=i * 50, dtype=np.uint8)
        write_image(input_path, data)

    print("--- Test environment setup complete ---")
    return num_images, image_shape

def cleanup_test_environment(test_dir: Path):
    """Removes all files and directories created during the test."""
    print(f"\n--- Cleaning up test environment in {test_dir} ---")
    if not test_dir.exists():
        return
    for item in test_dir.iterdir():
        if item.is_dir():
            cleanup_test_environment(item)
        else:
            item.unlink()
    test_dir.rmdir()
    print("--- Cleanup complete ---")


def main():
    test_dir = Path("./temp_pipeline_test")

    try:
        num_images, image_shape = setup_test_environment(test_dir)

        # --- Instantiate Controller ---
        controller = RawGLController(executable_path=str(test_dir / "rawgl.exe"))
        print(f"\nSuccessfully instantiated RawGLController.")

        # --- Create Jobs ---
        jobs = []
        for i in range(num_images):
            jobs.append(RawGLJob(
                shader_path=test_dir / "invert.frag",
                input_path=test_dir / f"input_{i}.png",
                output_path=test_dir / f"output_{i}.png",
                pass_size=image_shape
            ))
        print(f"Created {len(jobs)} jobs for the pipeline.")

        # --- Mock the subprocess call ---
        def mock_run_single_job(self, args: List[str]):
            try:
                # The argument we are looking for is the path that follows "--out" and "out_color"
                output_arg_index = args.index('--out') + 2
                output_path = Path(args[output_arg_index])
                # Simulate a successful run by creating the output file
                output_path.touch()
                return (0, "Mock success", None)
            except (ValueError, IndexError) as e:
                # This will help debug if the arguments are not as expected
                return (1, None, f"Mock error: Could not find output path in args. {e}")

        RawGLController.run_single_job = mock_run_single_job
        print("\nMocked RawGLController.run_single_job to simulate success.")

        # --- Run Pipeline ---
        controller.run_pipeline(jobs, max_workers=2)

        # --- Verify Results ---
        print("\n--- Verifying pipeline outputs ---")
        all_found = True
        for job in jobs:
            if job.output_path.exists():
                print(f"  [OK] Output file found: {job.output_path}")
            else:
                print(f"  [FAIL] Output file MISSING: {job.output_path}")
                all_found = False

        if all_found:
            print("\n--- Integration Test PASSED ---")
        else:
            raise RuntimeError("Integration test FAILED: Not all output files were created.")

    except Exception as e:
        print(f"\n--- Integration Test FAILED ---")
        print(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        cleanup_test_environment(test_dir)

if __name__ == '__main__':
    main()
