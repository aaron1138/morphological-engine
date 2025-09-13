import argparse
import configparser
import os
import shutil
import subprocess
import sys
import numpy
import moderngl
from PIL import Image

def check_slangc():
    """Check if slangc executable is in the system's PATH."""
    if not shutil.which("slangc"):
        print("Error: slangc not found in PATH.", file=sys.stderr)
        print("Please install RetroArch or the slangc compiler and ensure it is in your system's PATH.", file=sys.stderr)
        sys.exit(1)

def compile_shader(slang_path, glsl_path, include_dir):
    """Compile a .slang shader to .glsl using slangc."""
    command = [
        "slangc",
        slang_path,
        "-I", include_dir,
        "-o", glsl_path,
        "-t", "glsl",
        "-p", "glsl_450",
        "-D", "FRAGMENT",
        "-e", "main"
    ]
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error compiling shader: {slang_path}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

def process_image(input_image_path, shader_preset_path, output_image_path):
    """The main image processing function."""
    ctx = None
    try:
        # --- 1. Setup and Parsing ---
        preset_dir = os.path.dirname(shader_preset_path)

        config = configparser.ConfigParser(interpolation=None)
        # Workaround for .slangp files that have no section headers
        with open(shader_preset_path, 'r') as f:
            content = "[DEFAULT]\n" + f.read()
        config.read_string(content)

        num_passes = int(config.get("DEFAULT", "shaders", fallback="0"))
        if num_passes == 0:
            print("Warning: Shader preset has no passes. Copying input to output.", file=sys.stderr)
            shutil.copy(input_image_path, output_image_path)
            return

        # --- 2. ModernGL and Image Loading ---
        print("Initializing headless GL context...")
        ctx = moderngl.create_standalone_context(require=450)

        print(f"Loading input image: {input_image_path}")
        img = Image.open(input_image_path).convert("RGBA")
        original_size = img.size

        # --- 3. Generic Vertex Shader and Quad Data ---
        vertex_shader_src = """
            #version 450
            in vec2 in_vert;
            out vec2 v_tex;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                v_tex = in_vert * 0.5 + 0.5;
            }
        """
        quad_buffer = ctx.buffer(numpy.array([
            -1.0, -1.0,  -1.0, 1.0,  1.0, -1.0,
            1.0, -1.0,   -1.0, 1.0,  1.0, 1.0,
        ], dtype='f4'))

        # --- 4. Main Processing Loop ---
        current_texture = ctx.texture(original_size, 4, img.tobytes())
        source_size = original_size

        # List to hold resources for cleanup
        resources = [current_texture, quad_buffer]

        print(f"Starting {num_passes}-pass shader pipeline...")
        for i in range(num_passes):
            print(f"--- Pass {i} ---")

            # --- 4a. Get Pass-Specific Shader Info ---
            shader_path_rel = config.get("DEFAULT", f"shader{i}")
            slang_path = os.path.join(preset_dir, shader_path_rel)
            glsl_path = slang_path + ".glsl"

            print(f"Compiling {slang_path}...")
            compile_shader(slang_path, glsl_path, preset_dir)

            with open(glsl_path, "r") as f:
                fragment_shader_src = f.read()
            os.remove(glsl_path) # Clean up temporary file

            prog = ctx.program(vertex_shader=vertex_shader_src, fragment_shader=fragment_shader_src)
            resources.append(prog)

            # --- 4b. Calculate Output Size for this Pass ---
            scale_type_x = config.get("DEFAULT", f"scale_type_x{i}", fallback="source")
            scale_type_y = config.get("DEFAULT", f"scale_type_y{i}", fallback="source")
            scale_x = float(config.get("DEFAULT", f"scale_x{i}", fallback="1.0"))
            scale_y = float(config.get("DEFAULT", f"scale_y{i}", fallback="1.0"))

            output_size = [0, 0]
            for j, (stype, scale) in enumerate([(scale_type_x, scale_x), (scale_type_y, scale_y)]):
                if stype == "source":
                    output_size[j] = int(source_size[j] * scale)
                elif stype == "viewport":
                    output_size[j] = int(original_size[j] * scale)
                elif stype == "absolute":
                    output_size[j] = int(scale)
                else: # Default to source
                    output_size[j] = int(source_size[j] * scale)
            output_size = tuple(output_size)

            print(f"Pass output size: {output_size}")

            # --- 4c. Create FBO and Uniforms ---
            next_texture = ctx.texture(output_size, 4)
            fbo = ctx.framebuffer(color_attachments=[next_texture])
            resources.extend([next_texture, fbo])

            fbo.use()
            ctx.viewport = (0, 0, output_size[0], output_size[1])

            # Uniforms
            if 'Source' in prog:
                filter_linear = config.getboolean("DEFAULT", f"filter_linear{i}", fallback=True)
                if filter_linear:
                    current_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                else:
                    current_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
                current_texture.use(0)
                prog['Source'].value = 0

            if 'UBO' in prog:
                # UBO layout: mat4 MVP, vec4 OutputSize, vec4 OriginalSize, vec4 SourceSize, vec4 FinalViewportSize, float FrameCount
                # We send data that slangc expects, even if our VS doesn't use all of it.
                mvp = numpy.identity(4, dtype='f4')

                ubo_data = numpy.zeros(16 + 4 + 4 + 4, dtype='f4')
                ubo_data[0:16] = mvp.flatten()
                ubo_data[16:20] = [output_size[0], output_size[1], 1.0/output_size[0], 1.0/output_size[1]]
                ubo_data[20:24] = [original_size[0], original_size[1], 1.0/original_size[0], 1.0/original_size[1]]
                ubo_data[24:28] = [source_size[0], source_size[1], 1.0/source_size[0], 1.0/source_size[1]]
                # The UBO in the shader has FrameCount as a member, but it's not in the std140 block.
                # The block binding is what matters. The size seems to be correct based on slang shader spec.

                ubo = ctx.buffer(data=ubo_data.tobytes())
                resources.append(ubo)
                # The binding point must match the `binding = X` in the shader (UBO binding=0).
                ubo.bind_to_uniform_block(binding=0)

            # --- 4d. Render ---
            render_obj = ctx.vertex_array(prog, [(quad_buffer, '2f', 'in_vert')])
            resources.append(render_obj)
            render_obj.render(moderngl.TRIANGLES)

            # --- 4e. Ping-Pong ---
            current_texture = next_texture
            source_size = output_size

        # --- 5. Save Output ---
        print("Processing finished. Reading final image from GPU...")
        fbo = ctx.framebuffer(color_attachments=[current_texture])
        raw_pixels = fbo.read(components=4, alignment=1, dtype='f4')

        # ModernGL returns raw bytes, need to convert to an image, accounting for float data
        # and vertical flip.
        img_out = Image.frombytes('RGBA', source_size, (numpy.frombuffer(raw_pixels, dtype='f4') * 255.0).astype('u1').tobytes())
        img_out = img_out.transpose(Image.FLIP_TOP_BOTTOM)

        print(f"Saving output to {output_image_path}")
        img_out.save(output_image_path, "PNG")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        # --- 6. Cleanup ---
        print("Releasing GPU resources...")
        if 'resources' in locals():
            for res in reversed(resources):
                res.release()
        if ctx:
            ctx.release()

def main():
    """Main function to parse arguments and run the shader tool."""
    parser = argparse.ArgumentParser(description="Apply RetroArch .slangp shader presets to images.")
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument("shader_preset", help="Path to the .slangp shader preset file.")
    parser.add_argument("output_image", help="Path to save the output PNG image.")

    args = parser.parse_args()

    check_slangc()

    process_image(args.input_image, args.shader_preset, args.output_image)


if __name__ == "__main__":
    main()
