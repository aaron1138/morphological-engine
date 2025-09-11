import sys
from pathlib import Path
import numpy as np
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.voxel_engine import VoxelEngine, SliceLoader
from pipeline.rawgl_controller import RawGLController
from data_io.image import write_image
from dask.distributed import Client, LocalCluster

def setup_test_environment(test_dir: Path):
    """Creates all necessary dummy files and directories for the test."""
    print(f"--- Setting up test environment in {test_dir} ---")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)

    (test_dir / "rawgl.exe").touch()
    (test_dir / "shader.frag").write_text("#version 330\n...")

    num_images = 20
    image_shape = (64, 64)
    for i in range(num_images):
        write_image(test_dir / f"slice_{i}.png", np.full(image_shape, i, dtype=np.uint8))

    print("--- Test environment setup complete ---")
    return num_images, image_shape

# Create a mock controller for testing that doesn't use subprocess
class MockRawGLController(RawGLController):
    def _process_chunk_3d(self, chunk: np.ndarray, shader_path: str, uniforms: dict, block_info=None) -> np.ndarray:
        """
        Mocks the shader processing on a 3D chunk by inverting each slice.
        """
        processed_slices = [np.invert(chunk[i]) for i in range(chunk.shape[0])]
        return np.stack(processed_slices, axis=0)

def main():
    test_dir = Path("./temp_dask_workflow_test")
    client = None
    cluster = None

    try:
        num_images, image_shape = setup_test_environment(test_dir)

        # 1. Simulate UI Configuration
        config = { "dask_workers": 2 }
        print(f"\n--- Using simulated config: {config} ---")

        # 2. Initialize Dask Cluster
        cluster = LocalCluster(n_workers=config["dask_workers"], threads_per_worker=1)
        client = Client(cluster)
        print(f"Dask client ready: {client.dashboard_link}")

        # 3. Initialize Voxel Engine
        loader = SliceLoader(str(test_dir))
        engine = VoxelEngine(loader, chunk_size=(10, 64, 64))
        volume = engine.get_dask_array()
        print("VoxelEngine created a Dask array.")

        # 4. Initialize the Mock RawGL Controller
        controller = MockRawGLController(str(test_dir / "rawgl.exe"), temp_dir=str(test_dir / "processing"))
        print("Instantiated MockRawGLController for processing.")

        # 5. Run the processing pipeline
        processed_vols = controller.apply_shader_to_volume(volume, str(test_dir / "shader.frag"))
        print("Applied shader to all orthogonal views.")

        # 6. Compute and Verify
        print("\nComputing XY view result...")
        result_xy = processed_vols[0].compute()

        print(f"Result computed with shape: {result_xy.shape}")
        assert result_xy.shape == volume.shape
        assert result_xy[0, 0, 0] == np.invert(np.uint8(0))

        print("\n--- Dask Workflow Integration Test PASSED ---")

    except Exception as e:
        print(f"\n--- Dask Workflow Integration Test FAILED: {e} ---")
        sys.exit(1)
    finally:
        # --- Cleanup ---
        print("\n--- Cleaning up test environment ---")
        if client and not client.status == 'closed':
            client.close()
        if cluster and not cluster.status == 'closed':
            cluster.close()

        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("--- Cleanup complete ---")

if __name__ == '__main__':
    main()
