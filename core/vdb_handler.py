"""
Module for handling OpenVDB grid operations.

This module contains functions for creating, loading, saving,
and manipulating OpenVDB grids.

NOTE: This module requires the `pyopenvdb` library to be installed.
See docs/SETUP.md for more information.
"""

try:
    import pyopenvdb as vdb
except ImportError:
    print("WARNING: pyopenvdb is not installed. VDB functionality will not be available.")
    # Define a dummy class to avoid errors on import if pyopenvdb is not present
    class vdb:
        class FloatGrid:
            def __init__(self):
                self.name = ""

        @staticmethod
        def write(path, grids):
            print(f"MOCK VDB: Writing to {path} (not really).")
            pass

        @staticmethod
        def readAll(path):
            print(f"MOCK VDB: Reading from {path} (not really).")
            return []


def create_float_grid(name="density"):
    """
    Creates a new, empty OpenVDB FloatGrid.

    Args:
        name (str): The name to assign to the grid.

    Returns:
        A new vdb.FloatGrid object.
    """
    print(f"Creating VDB FloatGrid named '{name}'...")
    # In a real implementation, you would create a grid and perhaps populate it.
    grid = vdb.FloatGrid()
    grid.name = name
    return grid

def save_vdb(filepath, grids):
    """
    Saves a list of VDB grids to a .vdb file.

    Args:
        filepath (str): The path to save the file to.
        grids (list): A list of VDB grid objects to save.
    """
    print(f"Saving {len(grids)} VDB grids to {filepath}...")
    vdb.write(filepath, grids=grids)
    print("Save complete.")

def load_vdb(filepath):
    """
    Loads VDB grids from a .vdb file.

    Args:
        filepath (str): The path to the .vdb file.

    Returns:
        A list of grid objects from the file.
    """
    print(f"Loading VDB grids from {filepath}...")
    try:
        grids = vdb.readAll(filepath)
        print(f"Loaded {len(grids)} grids.")
        return grids
    except Exception as e:
        print(f"Error loading VDB file: {e}")
        return []
