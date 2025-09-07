import moderngl
import os
from utils import gl_utils

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720
OUTPUT_DIR = "output"
OUTPUT_FILENAME = "test_render.png"
SHADER_DIR = "shaders"
VERT_SHADER = "fullscreen.vert"
FRAG_SHADER = "test_pattern.frag"

def main():
    """
    Main function to run the ModernGL rendering test.
    """
    print("Starting ModernGL verification script...")

    # --- 1. Initialization ---
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    # Create a headless ModernGL context
    try:
        ctx = gl_utils.create_headless_context()
    except Exception as e:
        print(f"Failed to create ModernGL context. Aborting. Error: {e}")
        return

    # --- 2. Resource Loading ---
    print("Loading GPU resources...")
    # Load the shader program
    vert_shader_path = os.path.join(SHADER_DIR, VERT_SHADER)
    frag_shader_path = os.path.join(SHADER_DIR, FRAG_SHADER)
    try:
        program = gl_utils.load_shader_program(ctx, vert_shader_path, frag_shader_path)
    except FileNotFoundError as e:
        print(f"Failed to load shaders. Make sure they exist at the correct path. Error: {e}")
        return

    # Create the geometry for a fullscreen quad
    vao = gl_utils.create_fullscreen_quad(ctx)

    # Create a framebuffer to render into
    fbo, color_texture, _ = gl_utils.create_framebuffer(ctx, (WIDTH, HEIGHT))

    # --- 3. Rendering ---
    print("Executing render pass...")
    # Activate the framebuffer as the render target
    fbo.use()
    # Clear the framebuffer with a dark color
    ctx.clear(0.1, 0.1, 0.1, 1.0)
    # Activate the shader program
    # (No uniforms to set for this simple shader)
    # Render the quad
    vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4)

    # --- 4. Output ---
    print(f"Saving rendered image to {output_path}...")
    try:
        # Save the content of the framebuffer to a PNG file
        gl_utils.save_framebuffer_to_png(fbo, output_path)
    except Exception as e:
        print(f"Failed to save image. Error: {e}")
        return

    print("\nVerification script finished successfully!")
    print(f"Check the '{output_path}' file to see the result.")


if __name__ == "__main__":
    main()
