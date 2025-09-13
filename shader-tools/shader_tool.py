# Backend command-line tool for applying RetroArch .slangp shader presets.

import argparse
import configparser
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

# A generic, pass-through vertex shader. This is sufficient for rendering a full-screen quad.
PASSTHROUGH_VERTEX_SHADER = """
#version 450
out vec2 FragCoord;
void main() {
    // Hardcoded vertices for a full-screen quad
    float x = -1.0 + float((gl_VertexID & 1) << 2);
    float y = -1.0 + float((gl_VertexID & 2) << 1);
    gl_Position = vec4(x, y, 0.0, 1.0);
    FragCoord = gl_Position.xy * 0.5 + 0.5;
}
"""

def check_slangc():
    """Check if the slangc compiler is available in the system's PATH."""
    if not shutil.which("slangc"):
        print(
            "Error: slangc not found in PATH.",
            "Please install the RetroArch shader toolchain and ensure it's in your system's PATH.",
            file=sys.stderr
        )
        sys.exit(1)

def compile_slang_to_glsl(slang_path, temp_dir):
    """
    Compile a .slang file to a GLSL fragment shader using slangc.
    Returns the path to the compiled .glsl file.
    """
    print(f"Compiling {slang_path}...")
    glsl_path = temp_dir / f"{slang_path.stem}_{os.urandom(4).hex()}.glsl"
    include_dir = slang_path.parent

    # Detect which compilation flag to use by inspecting the shader content
    try:
        with open(slang_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except IOError as e:
        print(f"Error reading shader file {slang_path}: {e}", file=sys.stderr)
        return None

    stage_flag = []
    if "#pragma stage" in content:
        print("Shader uses '#pragma stage'. Compiling with '-stage frag'.")
        stage_flag = ["-stage", "frag"]
    else:
        print("Shader does not use '#pragma stage'. Compiling with '-D FRAGMENT'.")
        stage_flag = ["-D", "FRAGMENT"]

    command = [
        "slangc", str(slang_path),
        "-I", str(include_dir),
        "-o", str(glsl_path),
        "-target", "glsl",
        "-profile", "glsl_450",
        *stage_flag,  # Unpack the chosen flag and its value
        "-entry", "main"
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # slangc often prints warnings to stderr even on success, so log them to stdout for info.
        if result.stderr:
            print(f"slangc compilation messages for {slang_path}:\n{result.stderr}")
        print(f"Successfully compiled to {glsl_path}")
        return glsl_path
    except subprocess.CalledProcessError as e:
        print(f"Error compiling {slang_path} with slangc:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return None

def parse_slangp(preset_path):
    """
    Parse a .slangp preset file to get the shader pipeline details.
    Returns a list of dictionaries, where each dict represents a shader pass.
    """
    print(f"Parsing preset file: {preset_path}")
    config = configparser.ConfigParser(interpolation=None, allow_no_value=True)

    try:
        with open(preset_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Prepend [DEFAULT] header if missing, to satisfy configparser
        if not content.strip().startswith('['):
            content = '[DEFAULT]\n' + content

        config.read_string(content)
    except Exception as e:
        print(f"Could not read or parse preset file: {e}", file=sys.stderr)
        raise ValueError(f"Failed to process preset file: {preset_path}") from e

    default_section = config['DEFAULT']

    if "shaders" not in default_section:
        raise ValueError("Invalid .slangp preset: 'shaders' count not found.")

    num_passes = default_section.getint("shaders")
    passes = []

    for i in range(num_passes):
        # Construct keys for the current pass
        path_key = f"shader{i}"
        filter_key = f"filter_linear{i}"
        scale_type_x_key = f"scale_type_x{i}"
        scale_type_y_key = f"scale_type_y{i}"
        scale_x_key = f"scale_x{i}"
        scale_y_key = f"scale_y{i}"
        alias_key = f"alias{i}"

        # Get shader path
        shader_rel_path_str = default_section.get(path_key)
        if not shader_rel_path_str:
            raise ValueError(f"Path for shader pass {i} ('{path_key}') not found in preset.")

        rel_path = Path(shader_rel_path_str.strip('"'))
        shader_path = (preset_path.parent / rel_path).resolve()

        if not shader_path.exists():
            raise FileNotFoundError(f"Shader file not found for pass {i}: {shader_path}")

        pass_info = {
            "path": shader_path,
            "filter_linear": default_section.getboolean(filter_key, fallback=False),
            "scale_type_x": default_section.get(scale_type_x_key, fallback="source"),
            "scale_type_y": default_section.get(scale_type_y_key, fallback="source"),
            "scale_x": default_section.getfloat(scale_x_key, fallback=1.0),
            "scale_y": default_section.getfloat(scale_y_key, fallback=1.0),
            "alias": default_section.get(alias_key, fallback=f"Pass{i}"),
        }
        passes.append(pass_info)

    print(f"Found {num_passes} shader passes.")
    return passes

def get_pass_output_size(pass_info, source_size, original_size):
    """Calculate the output size for a given shader pass."""
    width, height = source_size

    # Calculate X dimension
    if pass_info["scale_type_x"] == "source":
        out_width = int(width * pass_info["scale_x"])
    elif pass_info["scale_type_x"] == "viewport":
        out_width = int(original_size[0] * pass_info["scale_x"])
    elif pass_info["scale_type_x"] == "absolute":
        out_width = int(pass_info["scale_x"])
    else:
        out_width = width

    # Calculate Y dimension
    if pass_info["scale_type_y"] == "source":
        out_height = int(height * pass_info["scale_y"])
    elif pass_info["scale_type_y"] == "viewport":
        out_height = int(original_size[1] * pass_info["scale_y"])
    elif pass_info["scale_type_y"] == "absolute":
        out_height = int(pass_info["scale_y"])
    else:
        out_height = height

    return max(1, out_width), max(1, out_height)


def process_image(input_path, preset_path, output_path):
    """The main image processing function."""
    check_slangc()

    # Use a temporary directory for compiled GLSL shaders
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        try:
            shader_passes = parse_slangp(preset_path)
        except (ValueError, FileNotFoundError, configparser.Error) as e:
            print(f"Error parsing preset file: {e}", file=sys.stderr)
            sys.exit(1)

        ctx = None
        resources = []
        try:
            # 1. Initialize headless ModernGL context
            ctx = moderngl.create_standalone_context(require=450)
            print("ModernGL context created.")

            # 2. Load input image
            input_image = Image.open(input_path).convert("RGBA")
            original_size = input_image.size
            print(f"Input image loaded: {input_path}, size: {original_size}")

            # 3. Create initial texture
            input_texture = ctx.texture(original_size, 4, input_image.tobytes())
            resources.append(input_texture)

            source_size = original_size
            current_texture = input_texture

            # 4. Multi-pass rendering pipeline
            for i, pass_info in enumerate(shader_passes):
                print(f"\n--- Processing Pass {i}: {pass_info['alias']} ---")

                # Compile .slang to .glsl
                glsl_path = compile_slang_to_glsl(pass_info["path"], temp_dir)
                if not glsl_path:
                    raise RuntimeError(f"Failed to compile shader for pass {i}")
                with open(glsl_path, 'r') as f:
                    fragment_shader = f.read()

                # Create shader program
                program = ctx.program(vertex_shader=PASSTHROUGH_VERTEX_SHADER, fragment_shader=fragment_shader)
                resources.append(program)
                print("Shader program created.")

                # Calculate output size for this pass
                output_size = get_pass_output_size(pass_info, source_size, original_size)
                print(f"Pass {i} SourceSize: {source_size}, OutputSize: {output_size}")

                # Create output framebuffer and texture ("ping-pong" buffer)
                output_texture = ctx.texture(output_size, 4)
                resources.append(output_texture)
                fbo = ctx.framebuffer(color_attachments=[output_texture])
                resources.append(fbo)

                # Set texture filtering for the input texture
                if pass_info["filter_linear"]:
                    current_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                else:
                    current_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

                # Set uniforms
                if "SourceSize" in program:
                    program["SourceSize"].value = source_size
                if "OriginalSize" in program:
                    program["OriginalSize"].value = original_size
                if "OutputSize" in program:
                    program["OutputSize"].value = output_size
                if "FrameCount" in program:
                    program["FrameCount"].value = 0 # Static image, so frame count is 0

                # Render a full-screen quad
                fbo.use()
                ctx.clear()
                current_texture.use(location=0) # Bind as texture unit 0
                program['Source'] = 0 # Tell shader to use texture unit 0
                ctx.render(moderngl.TRIANGLES, 3)

                # The output of this pass is the input for the next
                current_texture = output_texture
                source_size = output_size

            # 5. Read final image back from the GPU
            print("\n--- Finalizing ---")
            final_fbo = ctx.framebuffer(color_attachments=[current_texture])
            raw_data = final_fbo.read(components=4, alignment=1)
            final_image = Image.frombytes("RGBA", current_texture.size, raw_data)

            # OpenGL's origin is bottom-left, Pillow's is top-left. Flip it.
            final_image = final_image.transpose(Image.FLIP_TOP_BOTTOM)

            # 6. Save the output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_image.save(output_path, format="PNG")
            print(f"Successfully saved final image to: {output_path}")

        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            # 7. Release all GPU resources
            print("Releasing GPU resources...")
            for resource in reversed(resources):
                resource.release()
            if ctx:
                ctx.release()
            print("Cleanup complete.")


def main():
    parser = argparse.ArgumentParser(description="Apply RetroArch .slangp shader presets to images.")
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("shader_preset", help="Path to the .slangp shader preset file.")
    parser.add_argument("output_image", help="Path to save the output PNG file.")
    args = parser.parse_args()

    input_path = Path(args.input_image)
    preset_path = Path(args.shader_preset)
    output_path = Path(args.output_image)

    if not input_path.is_file():
        print(f"Error: Input image not found at {input_path}", file=sys.stderr)
        sys.exit(1)
    if not preset_path.is_file():
        print(f"Error: Shader preset not found at {preset_path}", file=sys.stderr)
        sys.exit(1)

    process_image(input_path, preset_path, output_path)

if __name__ == "__main__":
    main()
