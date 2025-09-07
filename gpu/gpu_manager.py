# -*- coding: utf-8 -*-
"""
Module: gpu_manager.py
Author: Gemini
Description: Provides a high-level interface for managing GPU contexts,
             resources (textures, framebuffers), and shader programs using ModernGL.
"""

import moderngl
from typing import List, Tuple, Dict, Optional

try:
    import pyopenvdb as vdb
except ImportError:
    vdb = None

class GpuManager:
    """
    Manages the GPU context and resources for headless compute operations.
    This class is intended to be used as a singleton in the application.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GpuManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initializes the GpuManager, creating a standalone ModernGL context.
        It specifically tries the 'egl' backend for headless environments.
        """
        if self._initialized:
            return

        try:
            # For headless/server environments, 'egl' is the preferred backend.
            self.ctx = moderngl.create_standalone_context(backend='egl')
            self._initialized = True
            self.resources: Dict[str, List[moderngl.Buffer | moderngl.Texture | moderngl.Framebuffer | moderngl.Program]] = {
                'programs': [],
                'textures': [],
                'framebuffers': [],
                'buffers': []
            }
            print(f"GpuManager initialized with backend: {self.ctx.info['GL_RENDERER']}")
        except Exception as e:
            print(f"EGL backend failed: {e}. Falling back to autodetect.")
            try:
                # Fallback for environments where EGL is not available (e.g., local desktop)
                self.ctx = moderngl.create_standalone_context()
                self._initialized = True
                self.resources: Dict[str, List[moderngl.Buffer | moderngl.Texture | moderngl.Framebuffer | moderngl.Program]] = {
                    'programs': [], 'textures': [], 'framebuffers': [], 'buffers': []
                }
                print(f"GpuManager initialized with fallback backend: {self.ctx.info['GL_RENDERER']}")
            except Exception as e2:
                print(f"Error initializing GpuManager with any backend: {e2}")
                self.ctx = None
                self._initialized = False

    def list_gpus(self) -> List[str]:
        """
        Lists available GPUs.

        NOTE: ModernGL (OpenGL) does not have a robust, cross-platform way to
        enumerate multiple GPUs. This function currently returns the name of the
        single GPU that the context was created on. True multi-GPU selection
        would require a different library (like PyCUDA or PyOpenCL) or
        platform-specific APIs (WMI on Windows).

        Returns:
            List[str]: A list containing the name of the active GPU renderer.
        """
        if not self.ctx:
            return ["No active GPU context"]
        return [self.ctx.info.get('GL_RENDERER', 'Unknown Renderer')]

    def get_active_gpu(self) -> str:
        """
        Gets the name of the GPU currently being used by the context.
        """
        return self.list_gpus()[0]

    def create_shader_program(self, vertex_shader: str, fragment_shader: str) -> Optional[moderngl.Program]:
        """
        Creates a shader program from vertex and fragment shader source code.

        Args:
            vertex_shader (str): The source code for the vertex shader.
            fragment_shader (str): The source code for the fragment shader.

        Returns:
            Optional[moderngl.Program]: The compiled shader program, or None on failure.
        """
        if not self.ctx: return None
        try:
            program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            self.resources['programs'].append(program)
            return program
        except Exception as e:
            print(f"Error compiling shader: {e}")
            return None

    def create_compute_shader(self, compute_shader: str) -> Optional[moderngl.ComputeShader]:
        """
        Creates a compute shader program from source code.

        Args:
            compute_shader (str): The source code for the compute shader.

        Returns:
            Optional[moderngl.ComputeShader]: The compiled compute shader, or None on failure.
        """
        if not self.ctx: return None
        try:
            shader = self.ctx.compute_shader(compute_shader)
            self.resources['programs'].append(shader)
            return shader
        except Exception as e:
            print(f"Error compiling compute shader: {e}")
            return None

    def create_texture3d(self, size: Tuple[int, int, int], components: int = 1, dtype: str = 'f4', data: Optional[bytes] = None) -> Optional[moderngl.Texture3D]:
        """
        Creates a 3D texture.

        Args:
            size (Tuple[int, int, int]): The (width, height, depth) of the texture.
            components (int): The number of components (e.g., 1 for R, 4 for RGBA).
            dtype (str): The data type of each component (e.g., 'f4' for float32).
            data (Optional[bytes]): The initial data to load into the texture.

        Returns:
            Optional[moderngl.Texture3D]: The created 3D texture, or None on failure.
        """
        if not self.ctx: return None
        try:
            texture = self.ctx.texture3d(size, components, data, dtype=dtype)
            self.resources['textures'].append(texture)
            return texture
        except Exception as e:
            print(f"Error creating 3D texture: {e}")
            return None

    def create_framebuffer(self, color_attachment: moderngl.Texture) -> Optional[moderngl.Framebuffer]:
        """
        Creates an offscreen framebuffer object (FBO).

        Args:
            color_attachment (moderngl.Texture): The texture to attach as the
                                                 color buffer.

        Returns:
            Optional[moderngl.Framebuffer]: The created framebuffer, or None on failure.
        """
        if not self.ctx: return None
        try:
            fbo = self.ctx.framebuffer(color_attachments=[color_attachment])
            self.resources['framebuffers'].append(fbo)
            return fbo
        except Exception as e:
            print(f"Error creating framebuffer: {e}")
            return None

    def nanovdb_from_openvdb(self, grid: 'vdb.Grid') -> Optional['vdb.NanoGrid']:
        """
        Converts an OpenVDB grid to a NanoVDB grid handle.

        This requires pyopenvdb to be compiled with NanoVDB support.

        Args:
            grid (vdb.Grid): The input OpenVDB grid.

        Returns:
            A NanoVDB grid handle, or None if conversion fails.
        """
        if vdb is None:
            print("Error: pyopenvdb is not installed.")
            return None
        try:
            # create_nano_grid is the function to convert an OpenVDB grid to NanoVDB
            handle = vdb.create_nano_grid(grid)
            return handle
        except Exception as e:
            print(f"Error converting OpenVDB to NanoVDB: {e}")
            return None

    def create_buffer_from_nanovdb(self, nanovdb_handle) -> Optional[moderngl.Buffer]:
        """
        Creates a ModernGL buffer from a NanoVDB grid handle's byte data.

        Args:
            nanovdb_handle: The NanoVDB grid handle returned by nanovdb_from_openvdb.

        Returns:
            A ModernGL buffer containing the NanoVDB data, or None on failure.
        """
        if not self.ctx: return None
        try:
            # The handle has a .buffer() method to get the raw data
            nanovdb_bytes = nanovdb_handle.buffer()
            buf = self.ctx.buffer(data=nanovdb_bytes)
            self.resources['buffers'].append(buf)
            return buf
        except Exception as e:
            print(f"Error creating buffer from NanoVDB: {e}")
            return None

    def cleanup(self):
        """
        Releases all tracked GPU resources and the context itself.
        """
        if not self._initialized or not self.ctx:
            return

        print("Cleaning up GPU resources...")
        for resource_type, resource_list in self.resources.items():
            for resource in resource_list:
                resource.release()
            print(f"  - Released {len(resource_list)} {resource_type}")

        self.ctx.release()
        print("GPU context released.")
        self._initialized = False
        GpuManager._instance = None

# --- Example Usage ---
if __name__ == '__main__':
    print("--- GpuManager Test ---")

    # Initialize the manager
    gpu_manager = GpuManager()

    if gpu_manager.ctx:
        # List GPUs
        print("\nAvailable GPUs:")
        for gpu_name in gpu_manager.list_gpus():
            print(f"  - {gpu_name}")

        # Create a 3D Texture
        print("\nCreating a 3D Texture (64x64x64)...")
        tex_3d = gpu_manager.create_texture3d((64, 64, 64), components=1, dtype='f4')
        if tex_3d:
            print(f"  -> Success! Texture created with size {tex_3d.size}")

        # Create a simple shader program
        print("\nCreating a simple shader program...")
        vert_shader = """
            #version 330 core
            in vec2 in_vert;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        """
        frag_shader = """
            #version 330 core
            out vec4 f_color;
            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        """
        shader = gpu_manager.create_shader_program(vert_shader, frag_shader)
        if shader:
            print("  -> Success! Shader program compiled.")

        # Test NanoVDB conversion
        if vdb:
            print("\nTesting OpenVDB to NanoVDB conversion...")
            # Create a simple OpenVDB grid
            source_grid = vdb.FloatGrid()
            source_grid.fill(min_active=(1, 1, 1), max_active=(5, 5, 5), value=1.0)
            source_grid.name = 'test_grid'

            # Convert to NanoVDB
            nanovdb_handle = gpu_manager.nanovdb_from_openvdb(source_grid)
            if nanovdb_handle:
                print("  -> Success! Converted to NanoVDB handle.")

                # Create a buffer from the NanoVDB data
                nanovdb_buffer = gpu_manager.create_buffer_from_nanovdb(nanovdb_handle)
                if nanovdb_buffer:
                    print(f"  -> Success! Created ModernGL buffer with size {nanovdb_buffer.size} bytes.")
            else:
                print("  -> Failed to convert to NanoVDB.")
        else:
            print("\nSkipping NanoVDB test: pyopenvdb not installed.")

        # Cleanup
        print("\nCleaning up resources...")
        gpu_manager.cleanup()
        print("Cleanup complete.")
    else:
        print("\nCould not run test because GpuManager failed to initialize.")
        print("This may be expected in environments without a GPU or proper graphics drivers.")
