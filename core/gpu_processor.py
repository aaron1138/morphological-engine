# -*- coding: utf-8 -*-
"""
Module: gpu_processor.py
Author: Jules
Description: Encapsulates all GPU operations using ModernGL. This module handles
             context creation, resource management (shaders, textures, buffers),
             and provides an interface for GPU-based processing pipelines.
"""

import moderngl
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Assuming AppSettings will be available to be imported
# from utils.app_settings import AppSettings

class GpuProcessor:
    """
    Manages a ModernGL context and all related GPU resources.
    """

    def __init__(self, settings_manager=None):
        """
        Initializes the GpuProcessor and creates a ModernGL context.

        Args:
            settings_manager: An instance of AppSettings to get GPU preferences.
        """
        self.settings = settings_manager
        self.ctx = self._create_context()

        # Dictionaries to manage created resources
        self.programs: Dict[str, moderngl.Program] = {}
        self.textures: Dict[str, moderngl.Texture] = {}
        self.buffers: Dict[str, moderngl.Buffer] = {}
        self.framebuffers: Dict[str, moderngl.Framebuffer] = {}

        print(f"GpuProcessor initialized. Vendor: {self.ctx.info['GL_VENDOR']}, "
              f"Renderer: {self.ctx.info['GL_RENDERER']}")

    def _create_context(self) -> moderngl.Context:
        """Creates a ModernGL context based on application settings."""
        gpu_id = self.settings.get("gpu_device_id", -1) if self.settings else -1

        try:
            if gpu_id != -1:
                # Attempt to create a context for a specific device
                return moderngl.create_context(require=430, device_index=gpu_id)
            else:
                # Create a standalone context on the default device
                return moderngl.create_standalone_context(require=430)
        except Exception as e:
            print(f"Error creating ModernGL context: {e}")
            print("Please ensure your GPU drivers are up to date and support OpenGL 4.3+.")
            raise

    @staticmethod
    def list_gpus() -> List[Dict[str, Any]]:
        """
        Queries and returns a list of available GPU devices.

        Returns:
            A list of dictionaries, each describing a GPU.
        """
        devices = []
        try:
            # This function is part of the experimental 'query_tools'
            from moderngl_window.context.headless import query_devices
            for i, device in enumerate(query_devices()):
                devices.append({
                    "id": i,
                    "name": device.name,
                    "type": device.type,
                    "is_hardware": device.is_hardware
                })
        except ImportError:
            print("Warning: `moderngl-window` is not installed. GPU listing is unavailable.")
            print("Install it with: pip install moderngl-window")
        except Exception as e:
            print(f"Error querying GPU devices: {e}")
        return devices

    def load_shader_program(self, name: str, vertex_path: str, fragment_path: str = None, compute_path: str = None):
        """
        Loads, compiles, and links shaders into a program.

        Args:
            name (str): A unique name to store the program under.
            vertex_path (str): Path to the vertex shader file.
            fragment_path (str): Path to the fragment shader file.
            compute_path (str): Path to a compute shader file.
        """
        if name in self.programs:
            print(f"Shader program '{name}' already loaded.")
            return self.programs[name]

        shader_sources = {}
        try:
            with open(vertex_path, 'r') as f:
                shader_sources['vertex_shader'] = f.read()
            if fragment_path:
                with open(fragment_path, 'r') as f:
                    shader_sources['fragment_shader'] = f.read()
            if compute_path:
                with open(compute_path, 'r') as f:
                    shader_sources['compute_shader'] = f.read()

            program = self.ctx.program(**shader_sources)
            self.programs[name] = program
            print(f"Shader program '{name}' loaded and compiled successfully.")
            return program
        except (IOError, moderngl.Error) as e:
            print(f"Error loading shader program '{name}': {e}")
            raise

    def create_texture_3d(self, name: str, size: Tuple[int, int, int], data: np.ndarray = None, dtype: str = 'f4') -> moderngl.Texture:
        """
        Creates a 3D texture and stores it in the manager.

        Args:
            name (str): A unique name to store and retrieve the texture.
            size (Tuple[int, int, int]): The (width, height, depth) of the texture.
            data (np.ndarray, optional): NumPy array to initialize the texture. Defaults to None.
            dtype (str, optional): Data type of the texture components (e.g., 'f4'). Defaults to 'f4'.

        Returns:
            moderngl.Texture: The created 3D texture.
        """
        if name in self.textures:
            # If texture of same name and size exists, just return it
            if self.textures[name].size == size:
                return self.textures[name]
            # Otherwise, release the old one before creating a new one
            self.textures[name].release()

        texture = self.ctx.texture3d(size, 4, dtype=dtype) # Assuming 4 components (RGBA) for now
        if data is not None:
            texture.write(data.astype(dtype))

        self.textures[name] = texture
        print(f"3D Texture '{name}' created with size {size}.")
        return texture

    def create_framebuffer(self, name: str, color_attachments: List[moderngl.Texture]) -> moderngl.Framebuffer:
        """
        Creates a framebuffer object with the given texture attachments.

        Args:
            name (str): A unique name for the framebuffer.
            color_attachments (List[moderngl.Texture]): A list of textures to attach.

        Returns:
            moderngl.Framebuffer: The created framebuffer object.
        """
        if name in self.framebuffers:
            self.framebuffers[name].release()

        fbo = self.ctx.framebuffer(color_attachments=color_attachments)
        self.framebuffers[name] = fbo
        print(f"Framebuffer '{name}' created.")
        return fbo

    def create_nanovdb_buffer(self, grid: Any) -> moderngl.Buffer:
        """
        (Placeholder) Converts an OpenVDB grid to a NanoVDB buffer on the GPU.

        This is a complex operation that requires a bridge between OpenVDB and
        the GPU memory space that ModernGL can access. The actual implementation
        will likely involve CUDA-OpenGL interop or a similar technique.

        Args:
            grid: An OpenVDB grid object.

        Returns:
            A ModernGL buffer containing the NanoVDB data.
        """
        print("Warning: `create_nanovdb_buffer` is a placeholder and not yet implemented.")
        # 1. Use pyopenvdb's tools to convert the OpenVDB grid to a NanoVDB buffer (in CPU memory).
        # 2. Create a ModernGL buffer with the same size.
        # 3. Write the NanoVDB CPU buffer data into the ModernGL buffer.
        # This approach avoids complex interop but involves a CPU-GPU transfer.
        # A more advanced solution would create the buffer directly on the GPU.

        # Placeholder returns an empty buffer
        dummy_buffer = self.ctx.buffer(reserve=1024) # reserve 1KB
        self.buffers["nanovdb_placeholder"] = dummy_buffer
        return dummy_buffer

    def destroy(self):
        """Releases all managed GPU resources."""
        print("Releasing GPU resources...")
        for program in self.programs.values():
            program.release()
        for texture in self.textures.values():
            texture.release()
        for buffer in self.buffers.values():
            buffer.release()
        for framebuffer in self.framebuffers.values():
            framebuffer.release()

        if self.ctx:
            self.ctx.release()

        print("GPU resources released.")
