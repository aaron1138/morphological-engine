import subprocess
from pathlib import Path
import numpy as np
import dask.array as da
import uuid
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_io.image import write_image, read_image

class RawGLController:
    """
    A controller to manage and execute processing pipelines on Dask arrays
    using the RawGL command-line tool.
    """
    def __init__(self, executable_path: str, temp_dir: str = "temp_rawgl_processing"):
        self.executable_path = Path(executable_path)
        if not self.executable_path.is_file():
            raise FileNotFoundError(f"RawGL executable not found at: {self.executable_path}")

        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def _process_chunk_3d(self, chunk: np.ndarray, shader_path: str, uniforms: dict, block_info=None) -> np.ndarray:
        """
        Processes a 3D NumPy array (a chunk from Dask) with RawGL by
        processing each 2D slice within the chunk.
        """
        processed_slices = []
        # The chunk is 3D, iterate over its first dimension (the slices)
        for i in range(chunk.shape[0]):
            slice_2d = chunk[i, :, :]
            unique_id = uuid.uuid4()
            input_path = self.temp_dir / f"input_{unique_id}.png"
            output_path = self.temp_dir / f"output_{unique_id}.png"

            try:
                write_image(input_path, slice_2d)

                pass_size = (slice_2d.shape[1], slice_2d.shape[0])
                args = self._build_cli_args(Path(shader_path), input_path, output_path, pass_size, uniforms)

                subprocess.run(args, check=True, capture_output=True, text=True)

                processed_slice = read_image(output_path)
                processed_slices.append(processed_slice)

            except Exception as e:
                print(f"Error processing slice in chunk: {e}")
                # Append a black slice on error
                processed_slices.append(np.zeros_like(slice_2d))
            finally:
                if input_path.exists():
                    input_path.unlink()
                if output_path.exists():
                    output_path.unlink()

        return np.stack(processed_slices, axis=0)

    def _build_cli_args(self, shader_path, input_path, output_path, pass_size, uniforms) -> list:
        args = [
            str(self.executable_path), "-P", str(shader_path),
            "--pass_size", str(pass_size[0]), str(pass_size[1]),
            "--in", "in_texture", str(input_path),
            "--out", "out_color", str(output_path),
            "--out_format", "r8", "--out_channels", "1", "--out_bits", "8",
        ]
        if uniforms:
            for name, value in uniforms.items():
                args.extend(["--in", name, str(value)])
        return args

    def apply_shader_to_volume(self, dask_volume: da.Array, shader_path: str, uniforms: dict = None) -> (da.Array, da.Array, da.Array):
        """
        Applies a GLSL shader to a 3D Dask array using map_blocks for parallel processing.
        """
        shader_path = Path(shader_path)
        if not shader_path.is_file():
            raise FileNotFoundError(f"Shader file not found: {shader_path}")

        print("--- Applying shader to XY, XZ, and YZ views using map_blocks ---")

        # Process XY view (original orientation)
        processed_xy = dask_volume.map_blocks(
            self._process_chunk_3d,
            shader_path=str(shader_path),
            uniforms=uniforms,
            dtype=dask_volume.dtype
        )

        # Process XZ view
        xz_view = dask_volume.transpose((1, 0, 2))
        processed_xz = xz_view.map_blocks(
            self._process_chunk_3d,
            shader_path=str(shader_path),
            uniforms=uniforms,
            dtype=xz_view.dtype
        ).transpose((1, 0, 2))

        # Process YZ view
        yz_view = dask_volume.transpose((2, 1, 0))
        processed_yz = yz_view.map_blocks(
            self._process_chunk_3d,
            shader_path=str(shader_path),
            uniforms=uniforms,
            dtype=yz_view.dtype
        ).transpose((2, 1, 0))

        return processed_xy, processed_xz, processed_yz

if __name__ == '__main__':
    # The main test is now in test_dask_workflow.py
    print("This module is not meant to be run directly.")
    print("Please run the integration test: python test_dask_workflow.py")
