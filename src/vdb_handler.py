"""
Module for handling OpenVDB grid operations.

This module provides functions for creating, reading, writing, and converting
OpenVDB grids. It relies on the 'pyopenvdb' library, which must be built
from source and installed in the Python environment as per the SETUP.md guide.
"""

import numpy as np

# This import will only work after OpenVDB with Python bindings is built and
# the PYTHONPATH is set correctly.
try:
    import pyopenvdb as vdb
except ImportError:
    print("Warning: 'pyopenvdb' module not found.")
    print("Please ensure OpenVDB is compiled with Python bindings and that the")
    print("PYTHONPATH is set correctly as described in docs/SETUP.md.")
    # Create a dummy module to allow the rest of the code to be parsed.
    class DummyGrid:
        def __init__(self, name='dummy'):
            self.name = name
        @staticmethod
        def createLevelSetSphere(radius, center, voxelSize, halfWidth):
            return DummyGrid('sphere')
    class DummyVDB:
        FloatGrid = DummyGrid
        def write(self, *args): pass
        def read(self, *args): return DummyGrid()
        class nanovdb:
            @staticmethod
            def grid_to_buffer(grid): return b"dummy_buffer"
    vdb = DummyVDB()


def create_level_set_sphere(
    radius: float = 50.0,
    center: tuple = (0.0, 0.0, 0.0),
    voxel_size: float = 0.5,
    half_width: float = 2.0
) -> vdb.FloatGrid:
    """
    Creates a narrow-band level set representation of a sphere in a FloatGrid.

    This uses the efficient built-in OpenVDB function.

    Args:
        radius: The radius of the sphere.
        center: The (x, y, z) coordinates of the sphere's center.
        voxel_size: The size of a single voxel.
        half_width: The width of the narrow band around the surface.

    Returns:
        A pyopenvdb.FloatGrid containing the signed distance field of a sphere.
    """
    grid = vdb.FloatGrid.createLevelSetSphere(radius, center, voxel_size, half_width)
    grid.name = 'sphere'
    return grid


def write_grid(filepath: str, grid: vdb.FloatGrid):
    """
    Writes a VDB grid to a .vdb file.

    Args:
        filepath: The path to the output .vdb file.
        grid: The grid object to write.
    """
    if not filepath.endswith('.vdb'):
        filepath += '.vdb'
    print(f"Writing grid '{grid.name}' to {filepath}")
    vdb.write(filepath, grids=[grid])


def read_grid(filepath: str) -> vdb.FloatGrid:
    """
    Reads a VDB grid from a .vdb file.

    Args:
        filepath: The path to the input .vdb file.

    Returns:
        The first grid found in the file.
    """
    print(f"Reading grid from {filepath}")
    grids = vdb.readAll(filepath)
    return grids[0] if grids else None


def convert_to_nanovdb(grid: vdb.FloatGrid) -> bytes:
    """
    Converts an OpenVDB grid to a NanoVDB byte buffer.

    NOTE: The exact function name and API for this conversion needs to be
    verified after compiling OpenVDB. The function `pyopenvdb.nanovdb.grid_to_buffer`
    is a placeholder based on the likely API structure.

    Args:
        grid: The OpenVDB grid to convert.

    Returns:
        A byte buffer containing the NanoVDB grid representation.
    """
    print(f"Converting grid '{grid.name}' to NanoVDB buffer.")
    # This is a placeholder for the actual conversion function.
    # The actual function might be named differently, e.g.,
    # `vdb.export_to_nanovdb_buffer` or something similar.
    # We assume it returns a bytes-like object.
    nanovdb_buffer = vdb.nanovdb.grid_to_buffer(grid)
    return nanovdb_buffer
