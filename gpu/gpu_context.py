import moderngl
import numpy as np

class GPUContext:
    """
    Manages a ModernGL context and provides methods for creating GPU resources.
    """
    def __init__(self, backend='egl'):
        """
        Initializes a standalone ModernGL context.

        Args:
            backend (str): The backend to use for the context. 'egl' is a good
                           default for headless rendering on systems with drivers.
                           Falls back to 'standalone' if egl fails.
        """
        try:
            self.ctx = moderngl.create_standalone_context(backend=backend)
            print(f"Successfully created ModernGL context with '{backend}' backend.")
        except Exception as e:
            print(f"Warning: Could not create context with '{backend}' backend: {e}")
            try:
                self.ctx = moderngl.create_standalone_context()
                print("Falling back to default 'standalone' backend.")
            except Exception as e_fallback:
                raise RuntimeError(f"Failed to create any ModernGL standalone context. Error: {e_fallback}")

        print("--- GPU Context Info ---")
        print(f"Vendor: {self.ctx.info['GL_VENDOR']}")
        print(f"Renderer: {self.ctx.info['GL_RENDERER']}")
        print(f"Version: {self.ctx.info['GL_VERSION']}")
        print("------------------------")

    def create_texture(self, size: tuple, components: int = 4, dtype: str = 'f1', data=None):
        """Creates a 2D Texture."""
        return self.ctx.texture(size, components, data, dtype=dtype)

    def create_framebuffer(self, size: tuple, components: int = 4, dtype: str = 'f1'):
        """Creates a Framebuffer Object with a color texture attachment."""
        texture = self.create_texture(size, components, dtype)
        return self.ctx.framebuffer(color_attachments=[texture])

    def create_program(self, vertex_shader, fragment_shader):
        """Creates a shader program from vertex and fragment shader source."""
        return self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    def create_quad_program(self):
        """Creates a simple shader program to render a full-screen quad."""
        vertex_shader = """
            #version 330
            in vec2 in_vert;
            out vec2 v_text;
            void main() {
                v_text = in_vert * 0.5 + 0.5;
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        """
        fragment_shader = """
            #version 330
            uniform sampler2D u_texture;
            in vec2 v_text;
            out vec4 f_color;
            void main() {
                f_color = texture(u_texture, v_text);
            }
        """
        return self.create_program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    def create_vertex_array(self, program, buffer_info):
        """
        Creates a Vertex Array Object (VAO).

        Args:
            program: The shader program to associate with the VAO.
            buffer_info (list): A list of tuples, where each tuple contains
                                (buffer, format, attribute_name).
        """
        return self.ctx.vertex_array(program, buffer_info)

    def create_quad_vao(self, program):
        """Creates a simple VAO for rendering a full-screen quad."""
        quad_buffer = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype='f4')

        buffer = self.ctx.buffer(quad_buffer)
        return self.create_vertex_array(program, [(buffer, '2f', 'in_vert')])

    def release(self):
        """Releases the ModernGL context and its resources."""
        self.ctx.release()

if __name__ == '__main__':
    print("--- GPUContext Test ---")
    try:
        gpu = GPUContext()

        # Test resource creation
        tex = gpu.create_texture((128, 128))
        print(f"Created Texture: {tex}")

        fbo = gpu.create_framebuffer((128, 128))
        print(f"Created Framebuffer: {fbo}")

        prog = gpu.create_quad_program()
        print(f"Created Program: {prog}")

        vao = gpu.create_quad_vao(prog)
        print(f"Created VAO: {vao}")

        # Test rendering a simple scene
        fbo.use()
        gpu.ctx.clear(0.2, 0.4, 0.6)
        # A real application would bind a texture and render the quad here:
        # tex.use(0)
        # prog['u_texture'].value = 0
        # vao.render(moderngl.TRIANGLE_STRIP)

        print("\nGPUContext initialized and resources created successfully.")

        gpu.release()
        print("GPUContext released.")

    except Exception as e:
        print(f"An error occurred during GPUContext test: {e}")
