import openvdb
import numpy as np

class VoxelGrid:
    """
    A wrapper for OpenVDB FloatGrid providing an interface for voxel manipulation.
    The grid stores voxel data as 8-bit grayscale values (0-255), but uses
    OpenVDB's FloatGrid for internal storage to leverage its capabilities.
    """

    def __init__(self, grid: openvdb.FloatGrid = None):
        """
        Initializes the VoxelGrid.

        Args:
            grid (openvdb.FloatGrid, optional): An existing OpenVDB FloatGrid.
                                                If None, an empty grid is created.
                                                Defaults to None.
        """
        if grid:
            self.grid = grid
        else:
            self.grid = openvdb.FloatGrid()
            self.grid.name = "density"
            self.grid.grid_class = openvdb.GridClass.FOG_VOLUME

    @classmethod
    def from_file(cls, filepath: str):
        """
        Loads a VoxelGrid from a .vdb file.

        Args:
            filepath (str): The path to the .vdb file.

        Returns:
            VoxelGrid: A new VoxelGrid instance.
        """
        try:
            # readAll returns a tuple of grids. We need to find the one we want.
            grids = openvdb.readAll(filepath)
            for grid in grids:
                if isinstance(grid, openvdb.FloatGrid):
                    return cls(grid)
            raise ValueError("No FloatGrid found in the VDB file.")
        except (IOError, ValueError, RuntimeError) as e:
            print(f"Error loading VDB file: {e}")
            raise

    def save(self, filepath: str):
        """
        Saves the VoxelGrid to a .vdb file.

        Args:
            filepath (str): The path to save the .vdb file.
        """
        openvdb.write(filepath, grids=[self.grid])

    def set_voxel(self, x: int, y: int, z: int, value: float):
        """
        Sets the value of a single voxel.

        Args:
            x (int): The x-coordinate of the voxel.
            y (int): The y-coordinate of the voxel.
            z (int): The z-coordinate of the voxel.
            value (float): The value to set.
        """
        accessor = self.grid.get_accessor()
        accessor.set_value((x, y, z), value)

    def get_voxel(self, x: int, y: int, z: int) -> float:
        """
        Gets the value of a single voxel.

        Args:
            x (int): The x-coordinate of the voxel.
            y (int): The y-coordinate of the voxel.
            z (int): The z-coordinate of the voxel.

        Returns:
            float: The value of the voxel.
        """
        accessor = self.grid.get_accessor()
        return accessor.get_value((x, y, z))

    def to_numpy(self) -> tuple[np.ndarray, tuple[int, int, int]]:
        """
        Converts the dense region of the grid to a NumPy array.
        Warning: This can be memory-intensive for large grids.

        Returns:
            A tuple containing:
            - np.ndarray: A NumPy array representing the active voxel data.
            - tuple[int, int, int]: The minimum coordinate (offset) of the bounding box.
        """
        min_ijk, _ = self.grid.eval_active_voxel_bbox()

        if not self.grid.has_active_voxels():
            return np.array([]), (0, 0, 0)

        arr = self.grid.copy_to_dense()
        return arr, min_ijk

    @classmethod
    def from_numpy(cls, array: np.ndarray, background_value: float = 0.0):
        """
        Creates a VoxelGrid from a NumPy array.

        Args:
            array (np.ndarray): The NumPy array containing voxel data.
            background_value (float): The value to treat as inactive/background.
                                      Voxels with this value will not be stored explicitly.

        Returns:
            VoxelGrid: A new VoxelGrid instance.
        """
        grid = openvdb.FloatGrid()
        grid.copy_from_dense(array)
        grid.background = background_value
        grid.name = "density"
        grid.grid_class = openvdb.GridClass.FOG_VOLUME
        return cls(grid)
