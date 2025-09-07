"""
Module for GPU processing using ModernGL and NanoVDB.

This module defines the GPUContext class, which is responsible for:
- GPU selection and context creation.
- Managing offscreen framebuffers, textures, and shaders.
- Running processing tasks on the GPU.
- Transferring data to/from the GPU.

NOTE: This module requires `moderngl` and `py-nanovdb`.
See docs/SETUP.md for more information.
"""

try:
    import moderngl
except ImportError:
    print("WARNING: moderngl is not installed. ModernGL functionality will not be available.")
    moderngl = None

try:
    import py_nanovdb as nanovdb
except ImportError:
    print("WARNING: py-nanovdb is not installed. NanoVDB functionality will not be available.")
    nanovdb = None

import numpy as np

class GPUProcessor:
    """A wrapper for ModernGL and NanoVDB context and operations."""

    def __init__(self, backend='egl'):
        """
        Initializes the GPU processor.

        Args:
            backend (str): The backend to use for ModernGL ('headless', 'pygame', etc.).
        """
        self.ctx = None
        if moderngl:
            try:
                # EGL is a more direct way to get a headless context.
                self.ctx = moderngl.create_context(standalone=True, backend=backend)
                print("ModernGL context created successfully.")
                print(f"Vendor: {self.ctx.info['GL_VENDOR']}")
                print(f"Renderer: {self.ctx.info['GL_RENDERER']}")
            except Exception as e:
                print(f"Error creating ModernGL context: {e}")
        else:
            print("Cannot initialize GPUProcessor without moderngl.")

    def create_offscreen_buffer(self, size, components=4, dtype='f1'):
        """
        Creates an offscreen framebuffer and texture.

        Args:
            size (tuple): The (width, height) of the buffer.
            components (int): The number of components in the texture (e.g., 1 for R, 4 for RGBA).
            dtype (str): The data type of the texture elements (e.g., 'f1' for 8-bit float).

        Returns:
            A tuple of (Framebuffer, Texture) objects.
        """
        if not self.ctx:
            return None, None

        print(f"Creating offscreen buffer of size {size}...")
        texture = self.ctx.texture(size, components, dtype=dtype)
        fbo = self.ctx.framebuffer(color_attachments=[texture])
        print("Offscreen buffer created.")
        return fbo, texture

    def create_shader(self, vertex_shader, fragment_shader):
        """
        Creates a shader program from source strings.

        Args:
            vertex_shader (str): The source code for the vertex shader.
            fragment_shader (str): The source code for the fragment shader.

        Returns:
            A ModernGL Program object.
        """
        if not self.ctx:
            return None

        print("Compiling shader program...")
        try:
            program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader,
            )
            print("Shader program compiled successfully.")
            return program
        except Exception as e:
            print(f"Error compiling shader: {e}")
            return None

    def render(self, fbo, vao, mode):
        """
        Renders a VAO to a framebuffer.

        Args:
            fbo (moderngl.Framebuffer): The framebuffer to render to.
            vao (moderngl.VertexArray): The vertex array object to render.
            mode: The ModernGL render mode (e.g., moderngl.TRIANGLE_STRIP).
        """
        if not self.ctx:
            return

        # Activate the framebuffer for rendering
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        vao.render(mode)
        print("Render complete.")

    def read_output(self, fbo, components=4, dtype='f1'):
        """
        Reads the pixel data from a framebuffer.

        Args:
            fbo (moderngl.Framebuffer): The framebuffer to read from.
            components (int): The number of components.
            dtype (str): The data type.

        Returns:
            A numpy array containing the pixel data.
        """
        if not self.ctx:
            return None

        print("Reading output from framebuffer...")
        output = np.frombuffer(
            fbo.read(components=components, dtype=dtype),
            dtype=np.uint8 # Assuming 8-bit for now
        ).reshape((*fbo.size, components))
        print("Output read successfully.")
        return output

    def load_nanovdb_grid(self, filepath):
        """
        Loads a NanoVDB grid from a file.
        """
        if not nanovdb:
            print("Cannot load NanoVDB grid without py-nanovdb.")
            return None

        print(f"Loading NanoVDB grid from {filepath}...")
        try:
            grid = nanovdb.read_grid(filepath)
            print("NanoVDB grid loaded.")
            return grid
        except Exception as e:
            print(f"Error loading NanoVDB grid: {e}")
            return None

    def release(self):
        """Releases the ModernGL context."""
        if self.ctx:
            self.ctx.release()
            print("ModernGL context released.")
