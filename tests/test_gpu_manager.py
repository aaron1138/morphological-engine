import pytest
import numpy as np
import os

# Make the src directory available for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to be able to import these to define the tests, but they will fail
# in an environment without the correct setup.
try:
    from src.gpu.manager import GPUManager
    from src.core.voxel import VoxelGrid
    GPU_AVAILABLE = True
except (ImportError, Exception):
    # This will catch both missing modules (openvdb) and ModernGL context errors
    GPU_AVAILABLE = False

# Pytest marker to skip tests if GPU or necessary libraries are not available.
requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires a GPU with OpenGL 4.3+ and all compiled libraries (OpenVDB).")

@requires_gpu
class TestGPUManager:

    @pytest.fixture(scope="class")
    def gpu_manager(self):
        """Fixture to provide a GPUManager instance for the test class."""
        try:
            manager = GPUManager()
            yield manager
            # Teardown: explicitly release resources if needed, though __del__ should handle it
            manager.ctx.release()
        except Exception as e:
            pytest.fail(f"Failed to initialize GPUManager. This test requires a functioning GPU environment. Error: {e}")

    @pytest.fixture(scope="class")
    def sample_voxel_grid(self):
        """Provides a sample VoxelGrid for testing."""
        # Using a small array size suitable for testing, dimensions divisible by workgroup size
        arr = np.arange(0, 8*8*4, dtype=np.float32).reshape((4, 8, 8)) # depth, height, width
        return VoxelGrid.from_numpy(arr)

    def test_texture_io(self, gpu_manager, sample_voxel_grid):
        """Tests writing data to a texture and reading it back."""
        input_np, _ = sample_voxel_grid.to_numpy()
        # VoxelGrid numpy is (depth, height, width), ModernGL texture is (width, height, depth)
        tex_shape = (input_np.shape[2], input_np.shape[1], input_np.shape[0])

        input_texture = gpu_manager.create_texture_3d(tex_shape, data=input_np)

        output_np = gpu_manager.read_texture_to_numpy(input_texture)

        # Reshape the output from ModernGL's (w, h, d) back to numpy's (d, h, w)
        output_np_reshaped = output_np.reshape((tex_shape[2], tex_shape[1], tex_shape[0]))

        assert np.allclose(input_np, output_np_reshaped)

    def test_compute_shader_processing(self, gpu_manager, sample_voxel_grid):
        """Tests the full compute shader pipeline."""
        input_np, _ = sample_voxel_grid.to_numpy()
        size = (input_np.shape[2], input_np.shape[1], input_np.shape[0])
        # Work group size is (8, 8, 4) in the shader
        work_groups = (size[0] // 8, size[1] // 8, size[2] // 4)

        # 1. Create resources
        input_texture = gpu_manager.create_texture_3d(size, data=input_np, dtype='f4')
        output_texture = gpu_manager.create_texture_3d(size, dtype='f4')

        shader_src = gpu_manager.load_shader_from_file('shaders/simple_process.glsl')
        compute_shader = gpu_manager.create_compute_shader(shader_src)

        # 2. Bind resources
        input_texture.bind_to_image(0, read=True, write=False)
        output_texture.bind_to_image(1, read=False, write=True)

        # 3. Run shader
        add_value = 10.0
        gpu_manager.run_compute_shader(compute_shader, work_groups, uniforms={'add_value': add_value})

        # 4. Get result
        output_np = gpu_manager.read_texture_to_numpy(output_texture)
        output_np_reshaped = output_np.reshape((size[2], size[1], size[0]))

        # 5. Verify
        expected_np = input_np + add_value
        assert np.allclose(expected_np, output_np_reshaped)
