"""
Module for handling GPU processing tasks using ModernGL.

This module provides a `GPUProcessor` class that encapsulates the logic for
creating a headless OpenGL context, managing GPU resources like textures,
framebuffers, and shaders, and running rendering operations.
"""

from typing import Tuple
import moderngl
import numpy as np
from PIL import Image


class GPUProcessor:
    """A class to manage GPU resources and processing via ModernGL."""

    def __init__(self, gpu_id: int = 0):
        """
        Initializes the GPUProcessor and creates a headless ModernGL context.

        Args:
            gpu_id: The ID of the GPU to use. Currently, this is a placeholder
                    as ModernGL's standalone context typically uses the default.
        """
        try:
            # Create a headless (standalone) context. This doesn't require a window.
            self.ctx = moderngl.create_standalone_context()
            print("ModernGL context created successfully.")
            print(f"GPU Vendor: {self.ctx.info.get('GL_VENDOR', 'N/A')}")
            print(f"GPU Renderer: {self.ctx.info.get('GL_RENDERER', 'N/A')}")
        except Exception as e:
            print(f"Error creating ModernGL context: {e}")
            print("Please ensure you have a functioning graphics driver installed.")
            self.ctx = None
            return

        self.program = None
        self.vao = None

    def load_shader_program(self, vertex_shader_path: str, fragment_shader_path: str):
        """
        Loads, compiles, and links vertex and fragment shaders into a program.

        Args:
            vertex_shader_path: Path to the vertex shader source file.
            fragment_shader_path: Path to the fragment shader source file.
        """
        if not self.ctx:
            return

        try:
            with open(vertex_shader_path, 'r') as f:
                vertex_shader = f.read()
            with open(fragment_shader_path, 'r') as f:
                fragment_shader = f.read()

            self.program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            print(f"Shader program loaded successfully from {vertex_shader_path} and {fragment_shader_path}")
        except Exception as e:
            print(f"Error loading shader program: {e}")
            self.program = None

    def create_offscreen_buffer(self, size: Tuple[int, int], components: int = 4, dtype: str = 'f1'):
        """
        Creates an offscreen framebuffer object (FBO) with a color attachment.

        Args:
            size: The (width, height) of the buffer.
            components: The number of components in the color texture (e.g., 4 for RGBA).
            dtype: The data type of the texture elements (e.g., 'f1' for 8-bit float).

        Returns:
            A tuple containing the created (moderngl.Framebuffer, moderngl.Texture).
        """
        if not self.ctx:
            return None, None

        color_texture = self.ctx.texture(size, components, dtype=dtype)
        fbo = self.ctx.framebuffer(color_attachments=[color_texture])
        print(f"Created offscreen buffer of size {size}")
        return fbo, color_texture

    def create_fullscreen_quad(self):
        """
        Creates a simple Vertex Array Object (VAO) for rendering a fullscreen quad.
        This is used to drive the fragment shader over the entire output buffer.
        """
        if not self.ctx or not self.program:
            print("Cannot create VAO without context and shader program.")
            return

        # Simple quad covering the entire screen in normalized device coordinates
        quad_vertices = np.array([
            # x,    y,   u,   v
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype='f4')

        vbo = self.ctx.buffer(quad_vertices)
        # The content array describes how the VBO is structured.
        # '2f' for vertex position (x, y)
        # '2f' for texture coordinates (u, v)
        # 'in_vert' and 'in_uv' must match the 'in' variable names in the vertex shader.
        self.vao = self.ctx.vertex_array(
            self.program,
            [(vbo, '2f 2f', 'in_vert', 'in_uv')]
        )
        print("Fullscreen quad VAO created.")

    def render(self, target_fbo: moderngl.Framebuffer, texture_to_process: moderngl.Texture = None):
        """
        Renders the fullscreen quad to the target framebuffer.

        Args:
            target_fbo: The framebuffer to render into.
            texture_to_process: The input texture to be processed by the shader.
        """
        if not all([self.ctx, self.program, self.vao, target_fbo]):
            print("Cannot render, a required component is missing.")
            return

        target_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0) # Clear with black

        if texture_to_process:
            # Bind the texture to texture unit 0
            texture_to_process.use(location=0)
            # Tell the shader that the 'u_texture' uniform corresponds to texture unit 0
            self.program['u_texture'].value = 0

        self.vao.render(moderngl.TRIANGLE_STRIP)
        print("Render command issued.")

    def read_output_to_numpy(self, fbo: moderngl.Framebuffer) -> np.ndarray:
        """
        Reads the pixel data from a framebuffer's color attachment into a NumPy array.

        Args:
            fbo: The framebuffer to read from.

        Returns:
            A NumPy array containing the pixel data. The shape will be (height, width, components).
        """
        if not self.ctx or not fbo:
            return np.array([])

        size = fbo.size
        components = fbo.color_attachments[0].components

        data = fbo.read(components=components, attachment=0, dtype='f1')
        image = np.frombuffer(data, dtype='u1').reshape((*size, components)[::-1])
        return image

    def save_output_to_png(self, fbo: moderngl.Framebuffer, output_path: str):
        """
        Reads the output from a framebuffer and saves it as a PNG image.

        Args:
            fbo: The framebuffer to read from.
            output_path: The path to save the PNG file.
        """
        if not output_path.lower().endswith('.png'):
            output_path += '.png'

        numpy_array = self.read_output_to_numpy(fbo)

        if numpy_array.size == 0:
            print("Cannot save PNG, source numpy array is empty.")
            return

        # Convert to a format Pillow can use easily
        if numpy_array.shape[2] == 1: # Grayscale
            image = Image.fromarray(numpy_array[:, :, 0], 'L')
        else: # RGB or RGBA
            image = Image.fromarray(numpy_array, 'RGBA')

        image.save(output_path)
        print(f"Output saved to {output_path}")

    def destroy(self):
        """Releases all ModernGL resources."""
        if self.program: self.program.release()
        if self.vao: self.vao.release()
        # FBOs and Textures are managed by the main script and should be released there.
        if self.ctx: self.ctx.release()
        print("GPU resources released.")
