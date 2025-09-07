import moderngl
import numpy as np
from PIL import Image

class HeadlessRenderer:
    """
    A class to handle offscreen rendering and processing using ModernGL.
    This renderer is designed to apply shader effects to 2D data (NumPy arrays).
    """

    def __init__(self, width, height, shaders):
        """
        Initializes the headless renderer.

        Args:
            width (int): The width of the processing framebuffer.
            height (int): The height of the processing framebuffer.
            shaders (dict): A dictionary containing 'vertex' and 'fragment' shader source code.
        """
        self.width = width
        self.height = height

        try:
            # 1. Create a headless (standalone) context
            self.ctx = moderngl.create_standalone_context()
            print("ModernGL context created successfully.")

            # 2. Create the shader program
            self.program = self.ctx.program(
                vertex_shader=shaders['vertex'],
                fragment_shader=shaders['fragment']
            )
            print("Shader program created.")

            # 3. Create a fullscreen quad
            vertices = np.array([
                # x,    y,   u,   v
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                -1.0,  1.0, 0.0, 1.0,
                 1.0,  1.0, 1.0, 1.0,
            ], dtype='f4')

            self.vbo = self.ctx.buffer(vertices)
            self.vao = self.ctx.vertex_array(
                self.program,
                [(self.vbo, '2f 2f', 'in_vert', 'in_uv')]
            )

            # 4. Create the framebuffer object to render to
            self.texture = self.ctx.texture((self.width, self.height), 3, dtype='f4') # RGB float texture
            self.fbo = self.ctx.framebuffer(color_attachments=[self.texture])
            print("Framebuffer and texture created.")

        except Exception as e:
            print(f"Error initializing HeadlessRenderer: {e}")
            raise

    def process_image(self, numpy_array):
        """
        Processes a NumPy array by applying the loaded shader.

        Args:
            numpy_array (np.ndarray): A HxWxC NumPy array (float32, 0-1 range).

        Returns:
            np.ndarray: The processed NumPy array.
        """
        if numpy_array.shape[:2] != (self.height, self.width):
            raise ValueError("Input array dimensions do not match renderer dimensions.")

        # Bind the framebuffer
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0)

        # Load the numpy array into the texture
        self.texture.write(numpy_array.astype('f4').tobytes())
        self.texture.use(0)
        self.program['u_texture'] = 0

        # Render the quad
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # Read the result back from the framebuffer
        output_data = self.fbo.read(components=3, dtype='f4')
        output_array = np.frombuffer(output_data, dtype='f4').reshape((self.height, self.width, 3))

        return output_array

    def release(self):
        """Releases all OpenGL resources."""
        self.fbo.release()
        self.texture.release()
        self.program.release()
        self.vbo.release()
        self.vao.release()
        self.ctx.release()
        print("All ModernGL resources released.")

# --- Example Usage ---
def get_invert_shader():
    """Returns a simple shader that inverts colors."""
    return {
        "vertex": """
            #version 330
            in vec2 in_vert;
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                v_uv = in_uv;
            }
        """,
        "fragment": """
            #version 330
            uniform sampler2D u_texture;
            in vec2 v_uv;
            out vec4 f_color;
            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                f_color = vec4(1.0 - color, 1.0);
            }
        """
    }

if __name__ == '__main__':
    # This is a simple example of how to use the HeadlessRenderer.
    # It will only run if ModernGL and NumPy are correctly installed.

    print("Running GL utils example...")
    width, height = 256, 256

    # 1. Create a sample image using NumPy
    # A simple gradient from black to red
    image_data = np.zeros((height, width, 3), dtype='f4')
    image_data[:, :, 0] = np.linspace(0, 1, width)

    renderer = None
    try:
        # 2. Initialize the renderer with an invert shader
        shaders = get_invert_shader()
        renderer = HeadlessRenderer(width, height, shaders)

        # 3. Process the image
        processed_data = renderer.process_image(image_data)

        # 4. Save the original and processed images to disk
        # Convert float (0-1) to uint8 (0-255)
        original_img = Image.fromarray((image_data * 255).astype(np.uint8), 'RGB')
        original_img.save("original_gradient.png")
        print("Saved original_gradient.png")

        processed_img = Image.fromarray((processed_data * 255).astype(np.uint8), 'RGB')
        processed_img.save("processed_gradient.png")
        print("Saved processed_gradient.png")

        print("GL utils example finished successfully.")

    except Exception as e:
        print(f"An error occurred during the example run: {e}")

    finally:
        if renderer:
            renderer.release()
