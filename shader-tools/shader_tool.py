import argparse
import configparser
import numpy as np
import moderngl
from PIL import Image
import shutil
import subprocess
import sys
import os
import tempfile

VERTEX_SHADER = """
#version 450
in vec2 in_vert;
out vec2 v_tex_coord;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_tex_coord = in_vert * 0.5 + 0.5;
}
"""

def check_slangc():
    """Check if slangc compiler is in the system's PATH."""
    if not shutil.which("slangc"):
        print("Error: slangc not found in PATH.", file=sys.stderr)
        print("Please install slangc and ensure it is in your system's PATH.", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to run the shader processing."""
    parser = argparse.ArgumentParser(description="Apply RetroArch .slangp shader presets to images.")
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument("shader_preset", help="Path to the .slangp shader preset.")
    parser.add_argument("output_image", help="Path to save the output image.")
    args = parser.parse_args()

    check_slangc()

    print(f"Input image: {args.input_image}")
    print(f"Shader preset: {args.shader_preset}")
    print(f"Output image: {args.output_image}")

    preset = parse_preset(args.shader_preset)
    print("Parsed preset:")
    import json
    print(json.dumps(preset, indent=4))

    include_dir = os.path.dirname(args.shader_preset)
    compiled_shaders = []
    temp_dirs = []

    try:
        for shader in preset["shaders"]:
            slang_path = os.path.join(include_dir, shader["path"])
            glsl_path = compile_shader(slang_path, include_dir)
            compiled_shaders.append(glsl_path)
            temp_dirs.append(os.path.dirname(glsl_path))

        # GPU processing
        process_image_with_shaders(args.input_image, compiled_shaders, preset, args.output_image)

    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir)

    print("Processing complete.")

def process_image_with_shaders(input_path, compiled_shaders, preset, output_path):
    """Process an image through the shader pipeline."""
    ctx = moderngl.create_standalone_context()

    input_image = Image.open(input_path).convert("RGBA")
    input_texture = ctx.texture(input_image.size, 4, input_image.tobytes())

    # Full screen quad
    vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype='f4')
    vbo = ctx.buffer(vertices)

    # UBO for shader parameters
    ubo_data = np.zeros(16, dtype='f4')
    ubo = ctx.buffer(data=ubo_data.tobytes(), dynamic=True)

    # Full screen quad
    vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype='f4')
    vbo = ctx.buffer(vertices)

    pass_textures = [input_texture]
    pass_fbo = []

    try:
        for i, shader_info in enumerate(preset["shaders"]):
            print(f"Processing pass {i+1}/{preset['num_shaders']}")

            # Calculate output size for this pass
            output_size = get_output_size(shader_info, input_image.size, pass_textures[-1].size)

            fbo = ctx.framebuffer(color_attachments=[ctx.texture(output_size, 4)])
            pass_fbo.append(fbo)
            fbo.use()

            with open(compiled_shaders[i], 'r') as f:
                fragment_shader = f.read()

            prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=fragment_shader)

            # Update UBO
            ubo_data[0:2] = pass_textures[-1].size  # SourceSize
            ubo_data[4:6] = input_image.size      # OriginalSize
            ubo_data[8:10] = output_size          # OutputSize
            ubo_data[12] = 0                      # FrameCount
            ubo.write(ubo_data.tobytes())

            ubo.bind_to_uniform_block(0)
            prog['ubo_params'].binding = 0

            vao = ctx.vertex_array(prog, [(vbo, '2f', 'in_vert')])

            pass_textures[-1].filter = (moderngl.LINEAR, moderngl.LINEAR) if shader_info["filter_linear"] else (moderngl.NEAREST, moderngl.NEAREST)
            pass_textures[-1].use(0)
            if "Source" in prog:
                prog["Source"].value = 0

            vao.render(moderngl.TRIANGLE_STRIP)

            pass_textures.append(fbo.color_attachments[0])

        # Read back the result
        final_fbo = pass_fbo[-1]
        final_fbo.use()

        data = final_fbo.read(components=4, alignment=1)
        result_image = Image.frombytes("RGBA", final_fbo.size, data).transpose(Image.FLIP_TOP_BOTTOM)
        result_image.save(output_path, "PNG")

    finally:
        for tex in pass_textures:
            tex.release()
        for fbo in pass_fbo:
            fbo.release()
        vbo.release()
        ubo.release()
        ctx.release()

def get_output_size(shader_info, original_size, source_size):
    """Calculate the output size for a shader pass."""
    scale_type_x = shader_info["scale_type_x"]
    scale_type_y = shader_info["scale_type_y"]
    scale_x = shader_info["scale_x"]
    scale_y = shader_info["scale_y"]

    if scale_type_x == "source":
        width = int(source_size[0] * scale_x)
    elif scale_type_x == "absolute":
        width = int(scale_x)
    else:  # viewport
        width = int(original_size[0] * scale_x)

    if scale_type_y == "source":
        height = int(source_size[1] * scale_y)
    elif scale_type_y == "absolute":
        height = int(scale_y)
    else:  # viewport
        height = int(original_size[1] * scale_y)

    return (width, height)

def compile_shader(slang_path, include_dir):
    """Compile a .slang shader to GLSL using slangc."""
    output_dir = tempfile.mkdtemp()
    glsl_path = os.path.join(output_dir, os.path.basename(slang_path) + ".glsl")

    command = [
        "slangc", slang_path,
        "-I", include_dir,
        "-target", "glsl",
        "-profile", "glsl_450",
        "-D", "FRAGMENT",
        "-entry", "main",
        "-o", glsl_path,
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Compiled {slang_path} successfully.")
        return glsl_path
    except subprocess.CalledProcessError as e:
        print(f"Error compiling {slang_path}:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        raise

def parse_preset(preset_path):
    """Parse a .slangp preset file."""
    config = configparser.ConfigParser(interpolation=None)
    config.read(preset_path)

    if not config.has_option("global", "shaders"):
        raise ValueError("Invalid preset: missing 'shaders' option in [global] section.")

    num_shaders = int(config["global"]["shaders"])
    shaders = []
    for i in range(num_shaders):
        shader_key = f"shader{i}"
        shader_path = config.get(shader_key, "shader", fallback=None)
        if not shader_path:
            raise ValueError(f"Missing 'shader' for {shader_key}")

        shaders.append({
            "path": shader_path,
            "filter_linear": config.getboolean(shader_key, "filter_linear", fallback=False),
            "scale_type_x": config.get(shader_key, "scale_type_x", fallback="source"),
            "scale_type_y": config.get(shader_key, "scale_type_y", fallback="source"),
            "scale_x": config.getfloat(shader_key, "scale_x", fallback=1.0),
            "scale_y": config.getfloat(shader_key, "scale_y", fallback=1.0),
        })
    return {"num_shaders": num_shaders, "shaders": shaders}

if __name__ == "__main__":
    main()
