import argparse
import configparser
import os
import shutil
import subprocess
import struct
import sys
import tempfile
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

# A generic, pass-through vertex shader for rendering a full-screen quad.
PASSTHROUGH_VERTEX_SHADER = """
#version 450
out vec2 FragCoord;
void main() {
    // Hardcoded vertices for a full-screen triangle
    float x = -1.0 + float((gl_VertexID & 1) << 2);
    float y = -1.0 + float((gl_VertexID & 2) << 1);
    gl_Position = vec4(x, y, 0.0, 1.0);
    FragCoord = (gl_Position.xy + 1.0) / 2.0;
}
"""

def check_slangc():
    """Check if the slangc compiler is available in the system's PATH."""
    if not shutil.which("slangc"):
        print("ERROR: slangc compiler not found in PATH.")
        print("Please install RetroArch and ensure its directory is in your system's PATH.")
        exit(1)

def parse_preset(preset_path):
    """Parse a .slangp file and return the shader pass configurations."""
    config = configparser.ConfigParser(interpolation=None)
    # RetroArch presets use '#' for comments and have other quirks
    with open(preset_path, 'r', encoding='utf-8') as f:
        content = f.read().replace('"', '') # Remove quotes for easier parsing
        # Add a dummy section header to make it a valid INI file for configparser
        config.read_string("[root]\n" + content)

    num_passes = config.getint('root', 'shaders')
    passes = []
    for i in range(num_passes):
        pass_data = {}
        shader_key = f'shader{i}'
        filter_key = f'filter_linear{i}'
        scale_type_x_key = f'scale_type_x{i}'
        scale_type_y_key = f'scale_type_y{i}'
        scale_x_key = f'scale_x{i}'
        scale_y_key = f'scale_y{i}'

        pass_data['path'] = Path(preset_path).parent / config.get('root', shader_key)
        pass_data['filter_linear'] = config.getboolean('root', filter_key, fallback=True)
        pass_data['scale_type_x'] = config.get('root', scale_type_x_key, fallback='source')
        pass_data['scale_type_y'] = config.get('root', scale_type_y_key, fallback='source')
        pass_data['scale_x'] = config.getfloat('root', scale_x_key, fallback=1.0)
        pass_data['scale_y'] = config.getfloat('root', scale_y_key, fallback=1.0)
        passes.append(pass_data)

    return passes

def compile_slang_to_glsl(slang_path, include_dir, temp_dir):
    """Compile a .slang file to a GLSL fragment shader using slangc."""
    output_path = temp_dir / f"{slang_path.stem}.glsl"
    command = [
        "slangc",
        "-I", str(include_dir),
        "-target", "glsl",
        "-profile", "glsl_450",
        "-stage", "fragment",
        "-entry", "main",
        "-o", str(output_path),
        str(slang_path)
    ]

    print(f"Compiling {slang_path.name}...")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("--- slangc COMPILE ERROR ---")
        print(f"Command: {' '.join(command)}")
        print(result.stderr)
        print("----------------------------")
        raise RuntimeError(f"slangc compilation failed for {slang_path}")

    with open(output_path, 'r') as f:
        return f.read()

def create_context_auto():
    """
    Automatically creates a headless moderngl context, providing detailed debug info.
    """
    print("--- Creating ModernGL Context ---")
    try:
        # Let moderngl try to find the best backend automatically.
        # It will try available backends like 'wgl', 'egl', 'glx', 'osmesa' etc.
        ctx = moderngl.create_context(standalone=True, require=450)
        print("Context created successfully.")
        print("  - Vendor: {}".format(ctx.info.get('GL_VENDOR', 'N/A')))
        print("  - Renderer: {}".format(ctx.info.get('GL_RENDERER', 'N/A')))
        print("  - Version: {}".format(ctx.info.get('GL_VERSION', 'N/A')))
        print("---------------------------------")
        return ctx
    except Exception as e:
        print("FATAL: Failed to create a headless ModernGL context.")
        print("This tool requires a functioning OpenGL 4.5 driver for headless processing.")
        print("On Windows, this is typically handled by the graphics driver (NVIDIA, AMD, Intel).")
        print("On Linux, you might need to install packages for EGL or OSMesa.")
        print(f"Underlying moderngl error: {e}")
        # Re-raise to ensure the program exits cleanly
        raise

def apply_shader(input_image_path, preset_path, output_image_path):
    """The main function to apply a shader preset to an image."""
    print(f"Processing {input_image_path} with {Path(preset_path).name}")

    ctx = None
    resources = []
    temp_dir = tempfile.TemporaryDirectory()

    try:
        # --- Initialization ---
        shader_passes = parse_preset(preset_path)
        preset_dir = Path(preset_path).parent

        # Create a headless context using the new helper function
        ctx = create_context_auto()

        # Load input image
        input_image = Image.open(input_image_path).convert("RGBA")
        original_size = input_image.size

        # Initial texture
        source_texture = ctx.texture(original_size, 4, input_image.tobytes())
        resources.append(source_texture)

        # UBO for shader parameters
        ubo_data = struct.pack('4f4f4f4f',
            *original_size, 0, 0, # SourceSize
            *original_size, 0, 0, # OriginalSize
            0, 0, 0, 0,            # OutputSize (will be updated per pass)
            0, 0, 0, 0             # FrameCount
        )
        uniform_buffer = ctx.buffer(data=ubo_data)
        resources.append(uniform_buffer)

        # The passthrough vertex shader is now passed directly to the program in the loop.
        # The VAO is also created per-pass, as it depends on the program.

        current_texture = source_texture
        current_size = original_size

        # --- Processing Passes ---
        for i, pass_info in enumerate(shader_passes):
            print(f"  - Pass {i+1}/{len(shader_passes)}: {pass_info['path'].name}")

            # Calculate output size for this pass
            if pass_info['scale_type_x'] == 'source':
                output_w = int(original_size[0] * pass_info['scale_x'])
            elif pass_info['scale_type_x'] == 'absolute':
                output_w = int(pass_info['scale_x'])
            else: # viewport, etc.
                output_w = int(current_size[0] * pass_info['scale_x'])

            if pass_info['scale_type_y'] == 'source':
                output_h = int(original_size[1] * pass_info['scale_y'])
            elif pass_info['scale_type_y'] == 'absolute':
                output_h = int(pass_info['scale_y'])
            else: # viewport, etc.
                output_h = int(current_size[1] * pass_info['scale_y'])

            output_size = (output_w, output_h)

            # Create framebuffer for this pass
            output_texture = ctx.texture(output_size, 4)
            fbo = ctx.framebuffer(color_attachments=[output_texture])
            resources.extend([output_texture, fbo])

            # Compile slang shader to GLSL
            fragment_glsl = compile_slang_to_glsl(pass_info['path'], preset_dir, Path(temp_dir.name))

            # Create program and VAO for this pass
            program = ctx.program(vertex_shader=PASSTHROUGH_VERTEX_SHADER, fragment_shader=fragment_glsl)
            vao = ctx.vertex_array(program, [])
            resources.extend([program, vao])

            # Update UBO
            ubo_update = struct.pack('4f4f4f',
                *current_size, 0, 0,  # SourceSize for this pass
                *original_size, 0, 0, # OriginalSize
                *output_size, 0, 0    # OutputSize for this pass
            )
            uniform_buffer.write(ubo_update, offset=0)

            # Bind resources
            fbo.use()
            ctx.viewport = (0, 0, *output_size)

            current_texture.use(location=0)
            if 'Original' in program:
                 source_texture.use(location=1) # Bind original image if shader needs it

            program['Source'] = 0
            if 'Original' in program:
                program['Original'] = 1

            uniform_buffer.bind_to_uniform_block(0)

            # Render
            vao.render(moderngl.TRIANGLES, vertices=3)

            # Ping-pong
            current_texture = output_texture
            current_size = output_size

        # --- Finalization ---
        # Read pixels from the last FBO
        final_fbo = fbo
        final_fbo.use()
        raw_pixels = final_fbo.read(components=4, alignment=1)

        # Create Pillow image from raw data
        result_image = Image.frombytes('RGBA', current_size, raw_pixels)
        result_image = result_image.transpose(Image.FLIP_TOP_BOTTOM)

        # Save the final image
        result_image.save(output_image_path, format='PNG')
        print(f"Successfully saved output to {output_image_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # --- Cleanup ---
        print("Releasing GPU resources...")
        for resource in resources:
            resource.release()
        if ctx:
            ctx.release()
        temp_dir.cleanup()
        print("Cleanup complete.")


def main():
    check_slangc()

    parser = argparse.ArgumentParser(description="Apply RetroArch .slangp shader presets to images.")
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("shader_preset", help="Path to the .slangp shader preset file.")
    parser.add_argument("output_image", help="Path to save the processed PNG image.")

    args = parser.parse_args()

    apply_shader(args.input_image, args.shader_preset, args.output_image)

if __name__ == "__main__":
    main()
