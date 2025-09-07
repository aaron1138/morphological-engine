import sys
import os
import pyopenvdb as vdb
# It is assumed the nanovdb bindings are part of the pyopenvdb module
# The exact import path might differ depending on the build configuration.
import pyopenvdb.nanovdb as nanovdb

# Add the src directory to the Python path to import the utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.vdb_utils import create_sphere_grid

def main():
    """
    Demonstrates the conversion of an OpenVDB grid to a NanoVDB grid.
    """
    print("--- Starting NanoVDB Conversion Example ---")

    # 1. Create a standard OpenVDB grid
    print("\nStep 1: Creating a standard OpenVDB grid...")
    sphere_grid = create_sphere_grid(radius=25.0, voxel_size=0.5)
    print(f"Created OpenVDB grid '{sphere_grid.name}' with {sphere_grid.activeVoxelCount()} active voxels.")

    # 2. Convert the OpenVDB grid to a NanoVDB grid
    print("\nStep 2: Converting to a NanoVDB grid...")
    nanovdb_handle = None
    try:
        # The function name is inferred from the C++ examples.
        # The actual Python binding might be named slightly differently,
        # e.g., createNanoGrid, create_grid, etc.
        nanovdb_handle = nanovdb.create_nano_grid(sphere_grid)

        if nanovdb_handle and not nanovdb_handle.is_null():
            grid_ptr = nanovdb_handle.grid()
            print("Successfully converted OpenVDB grid to NanoVDB.")
            print(f"NanoVDB grid class: {grid_ptr.grid_class()}")
            print(f"NanoVDB grid voxel count: {grid_ptr.voxel_count()}")

            # 3. Save the NanoVDB grid to a file
            output_path = "test_sphere.nvdb"
            print(f"\nStep 3: Saving NanoVDB grid to {output_path}...")
            nanovdb.io.write_grid(output_path, nanovdb_handle)
            print("Save complete.")

        else:
            print("Conversion failed: The returned NanoVDB handle is null.")

    except AttributeError:
        print("\nConversion failed: The function 'create_nano_grid' or module 'pyopenvdb.nanovdb' was not found.")
        print("This likely means OpenVDB was not compiled with the NanoVDB option enabled (-D OPENVDB_BUILD_NANOVDB=ON).")
    except Exception as e:
        print(f"\nAn unexpected error occurred during conversion: {e}")


    print("\n--- NanoVDB Conversion Example Finished ---")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"ImportError: {e}")
        print("\nPlease ensure pyopenvdb is installed and accessible.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
