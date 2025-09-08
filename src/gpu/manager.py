import moderngl
import numpy as np
from typing import Tuple, Optional

class GPUManager:
    """
    Manages GPU resources and processing tasks using ModernGL.
    This class handles the creation of a headless ModernGL context,
    as well as the management of textures, shaders, and other GPU objects.
    """
    def __init__(self):
        """
        Initializes the GPUManager and creates a headless ModernGL context.
        """
        self.ctx: Optional[moderngl.Context] = None
        try:
            # Create a standalone context for offscreen rendering/processing
            self.ctx = moderngl.create_standalone_context(require=430) # Require OpenGL 4.3 for compute shaders
            print("ModernGL context created successfully.")
            print(f"  Vendor: {self.ctx.info['GL_VENDOR']}")
            print(f"  Renderer: {self.ctx.info['GL_RENDERER']}")
            print(f"  OpenGL Version: {self.ctx.info['GL_VERSION']}")
        except Exception as e:
            print(f"Error: Could not create ModernGL context. {e}")
            print("Please ensure you have compatible OpenGL drivers installed (OpenGL 4.3+).")
            raise

    def create_texture_3d(self, size: Tuple[int, int, int], data: Optional[np.ndarray] = None, dtype: str = 'f4') -> moderngl.Texture:
        """
        Creates a 3D texture.

        Args:
            size (tuple): The (width, height, depth) of the texture.
            data (np.ndarray, optional): Data to upload to the texture. Defaults to None.
            dtype (str): The data type of the texture components (e.g., 'f4' for float32).

        Returns:
            A ModernGL 3D texture.
        """
        # For 8-bit grayscale, we might use 'f1' or 'u1', but 'f4' is better for compute
        components = 1
        if data is None:
            return self.ctx.texture3d(size, components, dtype=dtype)

        return self.ctx.texture3d(size, components, data.astype(dtype).tobytes(), dtype=dtype)

    def create_shader(self, vertex_src: str, fragment_src: str) -> moderngl.Program:
        """
        Creates a shader program from vertex and fragment shader source code.
        """
        if self.ctx is None:
            raise RuntimeError("ModernGL context is not available.")
        return self.ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)

    def create_compute_shader(self, compute_src: str) -> moderngl.ComputeShader:
        """
        Creates a compute shader program.
        """
        if self.ctx is None:
            raise RuntimeError("ModernGL context is not available.")
        return self.ctx.compute_shader(compute_src)

    def load_shader_from_file(self, filepath: str) -> str:
        """Loads shader source code from a file."""
        with open(filepath, 'r') as f:
            return f.read()

    def _convert_vdb_to_nanovdb_buffer(self, voxel_grid) -> bytes:
        """
        [PLACEHOLDER] Converts an OpenVDB grid to a NanoVDB byte buffer.

        This is a critical step that requires a custom C++ binding, as pyopenvdb
        does not expose the nanovdb::createNanoGrid() function directly.

        Args:
            voxel_grid: An instance of the VoxelGrid class from src.core.voxel.

        Returns:
            A bytes object containing the compact NanoVDB data structure.
        """
        print("Warning: _convert_vdb_to_nanovdb_buffer is a placeholder and does not perform a real conversion.")
        # In a real implementation, this would involve:
        # 1. Calling a custom C++ function that takes the openvdb.FloatGrid.
        # 2. In C++, using nanovdb::createNanoGrid() to get a handle.
        # 3. Accessing the raw buffer from the handle.
        # 4. Returning this buffer to Python as a bytes object.
        return b''

    def run_compute_shader(self, shader: moderngl.ComputeShader, work_groups: Tuple[int, int, int], uniforms: dict = None):
        """
        Runs a compute shader.

        Args:
            shader (moderngl.ComputeShader): The compute shader to run.
            work_groups (Tuple[int, int, int]): The number of work groups to launch.
            uniforms (dict, optional): A dictionary of uniforms to set.
        """
        if uniforms:
            for key, value in uniforms.items():
                if key in shader:
                    shader[key].value = value

        shader.run(group_x=work_groups[0], group_y=work_groups[1], group_z=work_groups[2])
        self.ctx.finish() # Wait for the compute shader to finish

    def read_texture_to_numpy(self, texture: moderngl.Texture) -> np.ndarray:
        """
        Reads the content of a texture back to a NumPy array on the CPU.

        Args:
            texture (moderngl.Texture): The texture to read from.

        Returns:
            A NumPy array with the texture's data.
        """
        buffer = self.ctx.buffer(reserve=texture.size)
        texture.read_into(buffer)

        # Determine numpy dtype from texture dtype
        if texture.dtype == 'f4':
            numpy_dtype = np.float32
        elif texture.dtype == 'f2':
            numpy_dtype = np.float16
        elif texture.dtype == 'u1':
            numpy_dtype = np.uint8
        else:
            # Default or raise error
            numpy_dtype = np.float32

        return np.frombuffer(buffer.read(), dtype=numpy_dtype).reshape(texture.shape)


    def __del__(self):
        """
        Clean up ModernGL resources upon object destruction.
        """
        if self.ctx:
            self.ctx.release()
            print("ModernGL context released.")
