import pyopenvdb as vdb
import numpy as np

def create_sphere_grid(radius: float, voxel_size: float = 0.1, center: tuple = (0.0, 0.0, 0.0)) -> vdb.FloatGrid:
    """
    Creates a new OpenVDB FloatGrid representing a level set sphere.

    Args:
        radius: The radius of the sphere in world units.
        voxel_size: The size of the voxels. A smaller value means higher resolution.
        center: The center of the sphere in world space.

    Returns:
        A new vdb.FloatGrid containing the sphere level set.
    """
    grid = vdb.createLevelSetSphere(radius=radius, center=center, voxelSize=voxel_size)
    grid.name = 'sphere'
    return grid

def save_vdb(filepath: str, grids: list):
    """
    Saves one or more OpenVDB grids to a .vdb file.

    Args:
        filepath: The path to the output .vdb file.
        grids: A list of OpenVDB grids to save.
    """
    if not filepath.endswith('.vdb'):
        filepath += '.vdb'

    try:
        vdb.write(filepath, grids=grids)
        print(f"Successfully saved grids to {filepath}")
    except Exception as e:
        print(f"Error saving VDB file: {e}")

def load_vdb(filepath: str) -> list:
    """
    Loads one or more grids from a .vdb file.

    Args:
        filepath: The path to the input .vdb file.

    Returns:
        A list of the OpenVDB grids contained in the file.
        Returns an empty list if the file cannot be read.
    """
    try:
        grids = vdb.readAll(filepath)
        print(f"Successfully loaded {len(grids)} grids from {filepath}")
        return grids
    except Exception as e:
        print(f"Error loading VDB file: {e}")
        return []

if __name__ == '__main__':
    # This is a simple example of how to use the utility functions.
    # It will only run if the OpenVDB library is correctly installed.

    print("Running VDB utils example...")

    # 1. Create a sphere grid.
    sphere_grid = create_sphere_grid(radius=50.0, voxel_size=0.5)
    print(f"Created a grid named '{sphere_grid.name}' of class {sphere_grid.gridClass}.")
    print(f"Active voxels: {sphere_grid.activeVoxelCount()}")

    # 2. Save the grid to a file.
    output_path = "test_sphere.vdb"
    save_vdb(output_path, [sphere_grid])

    # 3. Load the grid back from the file.
    loaded_grids = load_vdb(output_path)

    if loaded_grids:
        loaded_sphere = loaded_grids[0]
        print(f"Loaded a grid named '{loaded_sphere.name}' of class {loaded_sphere.gridClass}.")
        print(f"Active voxels: {loaded_sphere.activeVoxelCount()}")

        # Verification
        if loaded_sphere.activeVoxelCount() == sphere_grid.activeVoxelCount():
            print("Verification successful: Voxel counts match.")
        else:
            print("Verification failed: Voxel counts do not match.")

    print("VDB utils example finished.")
