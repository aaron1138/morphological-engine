#!/usr/bin/env python3

"""
RetroArch Shader Tool (Backend)

A command-line tool for applying RetroArch .slangp shader presets to static images.
"""

import argparse
import configparser
import logging
import numpy
import os
import shutil
import subprocess
import sys
import tempfile
from PIL import Image

# Attempt to import ModernGL, provide a helpful error message if it's not installed.
try:
    import ModernGL
except ImportError:
    print("Error: ModernGL is not installed. Please install it with 'pip install moderngl'.")
    sys.exit(1)

# --- Constants ---
PASS_THROUGH_VERTEX_SHADER = """
#version 450
// Simple pass-through vertex shader.
// Renders a full-screen quad.
vec2 vertices[4] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0)
);
out vec2 v_tex_coord;
void main() {
    gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
    v_tex_coord = (vertices[gl_VertexID] + 1.0) / 2.0;
}
"""

# --- Utility Functions ---

def check_slangc_availability():
    """
    Checks if the 'slangc' executable is in the system's PATH.
    Exits gracefully if it's not found.
    """
    if not shutil.which("slangc"):
        logging.error("Error: 'slangc' compiler not found in system PATH.")
        logging.error("Please ensure RetroArch is installed and its 'slangc' executable is accessible.")
        sys.exit(1)
    logging.info("'slangc' compiler found.")

def parse_shader_preset(preset_path):
    """
    Parses a .slangp (INI-style) shader preset file.

    Args:
        preset_path (str): Path to the .slangp file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a shader pass
              and contains its parameters.
    """
    logging.info(f"Parsing shader preset: {preset_path}")
    if not os.path.exists(preset_path):
        logging.error(f"Preset file not found: {preset_path}")
        sys.exit(1)

    parser = configparser.ConfigParser(interpolation=None)
    # configparser by default treats keys without values as lines to be ignored.
    # We also need to disable case-insensitivity for keys.
    parser.optionxform = str
    parser.read(preset_path)

    try:
        num_passes = int(parser.get('parameters', 'shaders'))
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        logging.error(f"Failed to read the number of shader passes from preset: {e}")
        logging.error("The .slangp file must have a 'shaders' key under a 'parameters' section or globally.")
        # Fallback for older presets where 'shaders' is a global key
        try:
            parser = configparser.ConfigParser(interpolation=None)
            parser.optionxform = str
            # Manually add a dummy section to read global keys
            with open(preset_path) as f:
                content = "[dummy_section]\n" + f.read()
            parser.read_string(content)
            num_passes = int(parser.get('dummy_section', 'shaders'))
        except (configparser.NoOptionError, ValueError) as fallback_e:
            logging.error(f"Fallback parsing failed: {fallback_e}")
            sys.exit(1)


    passes = []
    for i in range(num_passes):
        try:
            shader_path = parser.get('dummy_section', f'shader{i}').strip('"')
            pass_info = {
                'shader_path': shader_path,
                'filter_linear': parser.getboolean('dummy_section', f'filter_linear{i}', fallback=True),
                'scale_type_x': parser.get('dummy_section', f'scale_type_x{i}', fallback='source'),
                'scale_type_y': parser.get('dummy_section', f'scale_type_y{i}', fallback='source'),
                'scale_x': parser.getfloat('dummy_section', f'scale_x{i}', fallback=1.0),
                'scale_y': parser.getfloat('dummy_section', f'scale_y{i}', fallback=1.0),
            }
            passes.append(pass_info)
        except configparser.NoOptionError as e:
            logging.error(f"Missing parameter for pass {i} in preset file: {e}")
            sys.exit(1)

    logging.info(f"Successfully parsed {len(passes)} shader passes.")
    return passes

def compile_slang_to_glsl(slang_path, include_dir, temp_dir):
    """
    Compiles a .slang shader to a GLSL fragment shader using slangc.

    Args:
        slang_path (str): Path to the source .slang file.
        include_dir (str): The directory to use for includes (-I flag).
        temp_dir (str): The temporary directory to store the compiled GLSL file.

    Returns:
        str: The path to the compiled GLSL file, or None if compilation fails.
    """
    if not os.path.isabs(slang_path):
        slang_path = os.path.join(include_dir, slang_path)

    if not os.path.exists(slang_path):
        logging.error(f"Shader source file not found: {slang_path}")
        return None

    output_glsl_path = os.path.join(temp_dir, os.path.basename(slang_path) + '.glsl')

    logging.info(f"Compiling {os.path.basename(slang_path)} to GLSL...")

    command = [
        'slangc',
        slang_path,
        '-I', include_dir,
        '-target', 'glsl',
        '-profile', 'glsl_450',
        '-D', 'FRAGMENT',
        '-entry', 'main',
        '-o', output_glsl_path
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if result.stderr:
            logging.warning(f"slangc compilation produced warnings for {slang_path}:\n{result.stderr}")
        logging.info(f"Successfully compiled to {output_glsl_path}")
        return output_glsl_path
    except subprocess.CalledProcessError as e:
        logging.error(f"slangc compilation failed for {slang_path}.")
        logging.error(f"Command: {' '.join(command)}")
        logging.error(f"Exit Code: {e.returncode}")
        logging.error(f"Stderr:\n{e.stderr}")
        return None
    except FileNotFoundError:
        # This is another check, in case the initial one was bypassed.
        logging.error("Error: 'slangc' command not found. Is it in your PATH?")
        sys.exit(1)


def apply_shaders(input_image_path, output_image_path, shader_passes, preset_dir):
    """
    The core GPU processing pipeline. Applies the full shader chain to the input image.

    Args:
        input_image_path (str): Path to the source image.
        output_image_path (str): Path to save the final processed image.
        shader_passes (list): The list of parsed shader pass data.
        preset_dir (str): The directory of the .slangp preset, for resolving relative paths.
    """
    logging.info("Initializing GPU processing pipeline...")

    ctx = None
    temp_dir = tempfile.mkdtemp(prefix="shader-tool-")

    try:
        # --- 1. Initialization and Data Loading ---
        ctx = ModernGL.create_standalone_context(require=450)
        logging.info(f"ModernGL context created. Vendor: {ctx.info['GL_VENDOR']}, Renderer: {ctx.info['GL_RENDERER']}")

        input_image = Image.open(input_image_path).convert('RGBA')
        original_size = input_image.size
        logging.info(f"Loaded input image: {input_image_path} (Size: {original_size})")

        # --- 2. Compile all shaders first ---
        compiled_shaders = []
        for pass_info in shader_passes:
            glsl_path = compile_slang_to_glsl(pass_info['shader_path'], preset_dir, temp_dir)
            if not glsl_path:
                raise RuntimeError("Shader compilation failed. Aborting.")
            with open(glsl_path, 'r') as f:
                compiled_shaders.append(f.read())

        # --- 3. Setup GPU Resources ---
        # Vertex shader is constant for all passes
        vert_shader = PASS_THROUGH_VERTEX_SHADER

        # UBO for shader parameters
        ubo_data_struct = numpy.dtype([
            ('SourceSize', '2f4'),
            ('OriginalSize', '2f4'),
            ('OutputSize', '2f4'),
            ('FrameCount', 'i4'),
            ('padding', '3i4') # UBOs require specific alignment (std140)
        ])
        ubo = ctx.buffer(size=ubo_data_struct.itemsize)

        # Ping-pong buffers (textures and FBOs)
        textures = [None] * (len(shader_passes) + 1)
        fbos = [None] * len(shader_passes)

        # Initial texture
        textures[0] = ctx.texture(original_size, 4, input_image.tobytes())
        textures[0].filter = (ModernGL.LINEAR, ModernGL.LINEAR)
        textures[0].repeat_x = False
        textures[0].repeat_y = False

        # --- 4. Multi-pass Rendering Loop ---
        current_size = original_size
        for i, pass_info in enumerate(shader_passes):
            logging.info(f"--- Processing Pass {i+1}/{len(shader_passes)} ---")

            # a) Calculate output size for this pass
            output_size = list(current_size)
            for j, axis in enumerate(['x', 'y']):
                scale_type = pass_info[f'scale_type_{axis}']
                scale = pass_info[f'scale_{axis}']
                if scale_type == 'source':
                    output_size[j] = int(original_size[j] * scale)
                elif scale_type == 'viewport':
                    # For static images, viewport is same as current pass output
                    output_size[j] = int(output_size[j] * scale)
                elif scale_type == 'absolute':
                    output_size[j] = int(scale)
            output_size = tuple(max(1, s) for s in output_size)
            logging.info(f"Pass {i+1} | Input size: {current_size} -> Output size: {output_size}")

            # b) Create output texture and FBO
            textures[i+1] = ctx.texture(output_size, 4)
            fbos[i] = ctx.framebuffer(color_attachments=[textures[i+1]])
            fbos[i].use()
            ctx.viewport = (0, 0, output_size[0], output_size[1])

            # c) Compile GLSL program for the pass
            frag_shader = compiled_shaders[i]
            program = ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader)

            # d) Update UBO with current pass data
            ubo_content = numpy.array([(current_size, original_size, output_size, 0, (0,0,0))], dtype=ubo_data_struct)
            ubo.write(ubo_content)
            ubo.bind_to_uniform_block(0)

            # e) Set uniforms (texture samplers)
            textures[i].use(location=0) # Bind input texture from previous pass
            program['Source'].value = 0

            # f) Render a full-screen quad
            vao = ctx.vertex_array(program, [])
            vao.render(mode=ModernGL.TRIANGLE_STRIP, vertices=4)

            # g) Update current_size for the next iteration
            current_size = output_size

            # h) Release per-pass resources
            vao.release()
            program.release()

        # --- 5. Retrieve Final Image ---
        logging.info("All passes complete. Reading final image from GPU...")
        final_fbo = fbos[-1]
        final_fbo.use()

        # Ensure correct viewport for reading
        ctx.viewport = (0, 0, current_size[0], current_size[1])

        data = final_fbo.read(components=4, alignment=1)
        final_image = Image.frombytes('RGBA', current_size, data)
        final_image = final_image.transpose(Image.FLIP_TOP_BOTTOM)

        # --- 6. Save and Cleanup ---
        final_image.save(output_image_path, format='PNG')
        logging.info(f"Successfully saved processed image to: {output_image_path}")

    except Exception as e:
        logging.error(f"An error occurred during GPU processing: {e}", exc_info=True)
        # Re-raise to be caught by the main try...except block
        raise
    finally:
        # --- 7. Release all GPU resources ---
        logging.info("Releasing GPU resources...")
        for tex in textures:
            if tex: tex.release()
        for fbo in fbos:
            if fbo: fbo.release()
        if ubo: ubo.release()
        if ctx: ctx.release()

        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        logging.info("Cleanup complete.")


# --- Main Execution ---

def main():
    """Main function to parse arguments and run the tool."""
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(
        description="A command-line tool for applying RetroArch .slangp shader presets to static images.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("shader_preset", help="Path to the .slangp shader preset file.")
    parser.add_argument("output_image", help="Path to save the processed output PNG image.")

    args = parser.parse_args()

    try:
        # 1. Check for slangc dependency
        check_slangc_availability()

        # 2. Parse the shader preset file
        preset_path = os.path.abspath(args.shader_preset)
        preset_dir = os.path.dirname(preset_path)
        shader_passes = parse_shader_preset(preset_path)

        if not shader_passes:
            logging.error("No valid shader passes found in the preset. Aborting.")
            sys.exit(1)

        # 3. Run the main processing pipeline
        input_path = os.path.abspath(args.input_image)
        output_path = os.path.abspath(args.output_image)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        apply_shaders(input_path, output_path, shader_passes, preset_dir)

    except Exception as e:
        logging.error(f"A critical error occurred: {e}")
        sys.exit(1)

    logging.info("Processing finished successfully.")

if __name__ == "__main__":
    main()
