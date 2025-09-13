import argparse
import configparser
import os
import re
from pathlib import Path

import slangpy
import numpy as np
from PIL import Image

def parse_slang_parameters(shader_path):
    """Parse #pragma parameter lines from a .slang file."""
    params = {}
    param_regex = re.compile(r'#pragma\s+parameter\s+([A-Za-z0-9_]+)\s+"([^"]+)"\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)?')
    with open(shader_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = param_regex.match(line)
            if match:
                name, desc, default, min_val, max_val, step = match.groups()
                params[name] = float(default)
    return params

def parse_preset(preset_path):
    """Parse a .slangp file and return the shader pass configurations."""
    config = configparser.ConfigParser(interpolation=None)
    with open(preset_path, 'r', encoding='utf-8') as f:
        content = f.read().replace('"', '')
        config.read_string("[root]\n" + content)

    num_passes = config.getint('root', 'shaders')
    passes = []

    root_params = {k: v for k, v in config.items('root')}

    for i in range(num_passes):
        pass_data = {}
        shader_key = f'shader{i}'
        filter_key = f'filter_linear{i}'
        scale_type_x_key = f'scale_type_x{i}'
        scale_type_y_key = f'scale_type_y{i}'
        scale_x_key = f'scale_x{i}'
        scale_y_key = f'scale_y{i}'

        pass_data['path'] = Path(preset_path).parent / root_params.get(shader_key)
        pass_data['filter_linear'] = root_params.get(filter_key, 'true').lower() == 'true'
        pass_data['scale_type_x'] = root_params.get(scale_type_x_key, 'source')
        pass_data['scale_type_y'] = root_params.get(scale_type_y_key, 'source')
        pass_data['scale_x'] = float(root_params.get(scale_x_key, 1.0))
        pass_data['scale_y'] = float(root_params.get(scale_y_key, 1.0))

        # Collect other parameters for this pass
        pass_data['params'] = {}
        for key, value in root_params.items():
            if key.endswith(str(i)):
                param_name = key[:-1]
                pass_data['params'][param_name] = value
        passes.append(pass_data)

    # The preset can also contain global parameters for all passes.
    global_preset_params = {k: v for k, v in root_params.items() if not k[:-1].endswith(tuple(str(i) for i in range(num_passes)))}

    return passes, global_preset_params


def main():
    parser = argparse.ArgumentParser(description="Apply a .slangp shader preset to an image.")
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("shader_preset", help="Path to the .slangp shader preset file.")
    parser.add_argument("output_image", help="Path to save the processed PNG image.")
    args = parser.parse_args()

    try:
        device = slangpy.Device()
        if not device:
            print("Failed to create slang device.")
            return

        shader_passes, global_preset_params = parse_preset(args.shader_preset)

        img = Image.open(args.input_image).convert("RGBA")

        texture_loader = slangpy.TextureLoader(device)
        current_texture = texture_loader.load_texture(args.input_image)

        original_size = (img.width, img.height)
        current_size = original_size

        for i, pass_info in enumerate(shader_passes):
            print(f"  - Pass {i+1}/{len(shader_passes)}: {pass_info['path'].name}")

            if pass_info['scale_type_x'] == 'source':
                output_w = int(original_size[0] * pass_info['scale_x'])
            elif pass_info['scale_type_x'] == 'absolute':
                output_w = int(pass_info['scale_x'])
            else: # viewport
                output_w = int(current_size[0] * pass_info['scale_x'])

            if pass_info['scale_type_y'] == 'source':
                output_h = int(original_size[1] * pass_info['scale_y'])
            elif pass_info['scale_type_y'] == 'absolute':
                output_h = int(pass_info['scale_y'])
            else: # viewport
                output_h = int(current_size[1] * pass_info['scale_y'])

            output_size = (output_w, output_h)

            output_texture = device.create_texture(
                type=slangpy.TextureType.texture_2d,
                format=slangpy.Format.rgba8_unorm,
                width=output_size[0],
                height=output_size[1],
                usage=slangpy.TextureUsage.unordered_access | slangpy.TextureUsage.copy_source,
            )

            module = device.load_module(str(pass_info['path']))
            entry_point = module.entry_point("main")
            program = device.link_program(modules=[module], entry_points=[entry_point])
            kernel = device.create_compute_kernel(program)

            sampler = device.create_sampler()

            shader_params = parse_slang_parameters(pass_info['path'])
            # Override with preset parameters
            for key, value in global_preset_params.items():
                if key in shader_params:
                    shader_params[key] = float(value)
            for key, value in pass_info['params'].items():
                 if key in shader_params:
                    shader_params[key] = float(value)

            global_params = {
                "MVP": slangpy.math.float4x4.identity(),
                "OutputSize": slangpy.math.float4(output_size[0], output_size[1], 1.0/output_size[0], 1.0/output_size[1]),
                "OriginalSize": slangpy.math.float4(original_size[0], original_size[1], 1.0/original_size[0], 1.0/original_size[1]),
                "SourceSize": slangpy.math.float4(current_size[0], current_size[1], 1.0/current_size[0], 1.0/current_size[1]),
            }

            kernel.dispatch(
                thread_count=(output_size[0], output_size[1], 1),
                inputTexture=current_texture,
                outputTexture=output_texture,
                params=shader_params,
                globals=global_params,
                smp=sampler
            )

            current_texture = output_texture
            current_size = output_size

        readback_buffer = device.create_buffer(
            size=current_texture.desc.width * current_texture.desc.height * 4,
            memory_type=slangpy.MemoryType.read_back,
        )

        encoder = device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            dst=readback_buffer, dst_offset=0, dst_size=readback_buffer.size,
            dst_row_pitch=current_texture.desc.width * 4, src=current_texture, src_layer=0, src_mip=0,
        )
        cmd_buffer = encoder.finish()
        device.submit_command_buffer(cmd_buffer)
        device.wait_for_idle()

        output_data_bytes = readback_buffer.to_numpy()
        output_image = Image.fromarray(output_data_bytes.reshape((current_size[1], current_size[0], 4)), 'RGBA')

        output_image.save(args.output_image, format='PNG')
        print(f"Output image saved to {args.output_image}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
