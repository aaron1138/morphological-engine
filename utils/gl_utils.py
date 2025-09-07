import moderngl
import numpy as np
from PIL import Image

def create_headless_context(device_index=None):
    """
    Creates a headless ModernGL context using the EGL backend.

    Args:
        device_index (int, optional): The index of the GPU to use.
                                      If None, the system default is used.

    Returns:
        moderngl.Context: The created ModernGL context.

    Raises:
        Exception: If the context cannot be created.
    """
    try:
        # EGL is the standard for headless rendering on Linux
        context = moderngl.create_standalone_context(backend='egl', device_index=device_index)
        print(f"Successfully created headless ModernGL context.")
        print(f"  Vendor: {context.info['GL_VENDOR']}")
        print(f"  Renderer: {context.info['GL_RENDERER']}")
        print(f"  Version: {context.info['GL_VERSION']}")
        return context
    except Exception as e:
        print(f"Error creating headless context: {e}")
        # Fallback for systems without EGL (e.g., some virtual environments)
        try:
            print("Attempting to fall back to 'osmesa' backend...")
            context = moderngl.create_standalone_context(backend='osmesa')
            print("Successfully created headless context with OSMesa.")
            return context
        except Exception as e_osmesa:
            print(f"OSMesa fallback also failed: {e_osmesa}")
            raise

def load_shader_program(ctx, vert_path, frag_path):
    """
    Loads a vertex and fragment shader from files and creates a program.

    Args:
        ctx (moderngl.Context): The ModernGL context.
        vert_path (str): The file path to the vertex shader.
        frag_path (str): The file path to the fragment shader.

    Returns:
        moderngl.Program: The compiled shader program.
    """
    with open(vert_path, 'r') as f:
        vertex_shader = f.read()

    with open(frag_path, 'r') as f:
        fragment_shader = f.read()

    program = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
    )
    return program

def create_fullscreen_quad(ctx):
    """
    Creates a Vertex Array Object (VAO) for rendering a fullscreen quad.

    Returns:
        moderngl.VertexArray: The VAO for the fullscreen quad.
    """
    vertices = np.array([
        # x,    y
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype='f4')

    content = [(ctx.buffer(vertices), '2f', 'in_vert')]
    # No index buffer needed, we'll render with TRIANGLE_STRIP
    vao = ctx.vertex_array(ctx.program(vertex_shader='''
        #version 330
        in vec2 in_vert;
        out vec2 uv;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            uv = in_vert * 0.5 + 0.5;
        }
    ''', fragment_shader='''
        #version 330
        out vec4 f_color;
        void main() { f_color = vec4(0.0); }
    '''), content)

    return vao

def create_framebuffer(ctx, size, components=4, dtype='f1'):
    """
    Creates a framebuffer with color and depth texture attachments.

    Args:
        ctx (moderngl.Context): The ModernGL context.
        size (tuple): The (width, height) of the framebuffer.
        components (int): The number of components for the color texture (e.g., 4 for RGBA).
        dtype (str): The data type of the color texture pixels (e.g., 'f1' for 8-bit bytes).

    Returns:
        tuple: A tuple containing the framebuffer (moderngl.Framebuffer),
               the color texture (moderngl.Texture), and the depth texture
               (moderngl.Texture).
    """
    color_texture = ctx.texture(size, components, dtype=dtype)
    depth_texture = ctx.depth_texture(size)
    fbo = ctx.framebuffer(
        color_attachments=[color_texture],
        depth_attachment=depth_texture,
    )
    return fbo, color_texture, depth_texture

def save_framebuffer_to_png(fbo, output_path, components=4):
    """
    Reads the content of a framebuffer and saves it as a PNG image.

    Args:
        fbo (moderngl.Framebuffer): The framebuffer to read from.
        output_path (str): The path to save the PNG file to.
        components (int): The number of components to read (e.g., 4 for RGBA).
    """
    if components == 4:
        mode = 'RGBA'
    elif components == 3:
        mode = 'RGB'
    elif components == 1:
        mode = 'L' # Grayscale
    else:
        raise ValueError(f"Unsupported number of components: {components}")

    # Ensure the FBO is bound for reading
    fbo.use()

    # Read the pixels from the active color attachment
    # 'f1' dtype corresponds to unsigned 8-bit integers (bytes)
    raw_pixels = fbo.read(components=components, attachment=0, dtype='f1')

    # Create an image from the raw pixel data
    image = Image.frombytes(mode, fbo.size, raw_pixels)

    # The image will be upside down, so we need to flip it
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Save the image
    image.save(output_path)
    print(f"Successfully saved framebuffer to {output_path}")
