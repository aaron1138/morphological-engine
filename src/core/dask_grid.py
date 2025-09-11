import dask.array as da
import dask_image.imread
from pathlib import Path
from typing import Tuple

class DaskGrid:
    """
    Represents a 3D grid of data loaded from a stack of 2D images,
    powered by Dask for chunked, parallel processing.
    """
    def __init__(self, image_folder_path: str, chunk_size: Tuple[int, int, int] = (64, 64, 64)):
        """
        Initializes the DaskGrid by loading a stack of images from a folder.

        Args:
            image_folder_path (str): The path to the folder containing the image stack (e.g., PNGs).
            chunk_size (Tuple[int, int, int]): The desired chunk size for the Dask array.
        """
        self.folder_path = Path(image_folder_path)
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"The provided path is not a directory: {image_folder_path}")

        self.dask_array = self._load_image_stack(chunk_size)

    def _load_image_stack(self, chunk_size: Tuple[int, int, int]) -> da.Array:
        """
        Loads all images in the directory into a single 3D Dask array.
        The images are stacked along the first dimension (depth).
        """
        # Find all .png files and sort them naturally (e.g., 1.png, 2.png, ..., 10.png)
        image_paths = sorted(
            list(self.folder_path.glob('*.png')),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
        )

        if not image_paths:
            raise FileNotFoundError(f"No .png files found in directory: {self.folder_path}")

        # Use dask_image.imread to lazily load the stack of images
        # This creates a Dask array where each slice is a Dask delayed object
        array = dask_image.imread.imread(str(self.folder_path / '*.png'))

        # Rechunk the array to the desired 3D chunking scheme for efficient orthogonal slicing.
        # Note: The input array from imread is chunked like (1, H, W).
        # Rechunking is necessary for good performance on XZ/YZ slicing.
        return array.rechunk(chunks={0: chunk_size[0], 1: chunk_size[1], 2: chunk_size[2]})

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Returns the shape of the full 3D grid."""
        return self.dask_array.shape

    @property
    def chunks(self) -> Tuple[Tuple[int, ...], ...]:
        """Returns the chunking scheme of the Dask array."""
        return self.dask_array.chunksize

    def get_slice(self, plane: str, index: int) -> da.Array:
        """
        Extracts a 2D slice from one of the three orthogonal planes.

        Args:
            plane (str): The plane to slice from. Must be one of 'XY', 'XZ', 'YZ'.
            index (int): The index of the slice on that plane.

        Returns:
            A 2D Dask array representing the slice.
        """
        if plane.upper() == 'XY':
            # A slice along the Z axis (the original image stack)
            return self.dask_array[index, :, :]
        elif plane.upper() == 'XZ':
            # A slice along the Y axis
            return self.dask_array[:, index, :]
        elif plane.upper() == 'YZ':
            # A slice along the X axis
            return self.dask_array[:, :, index]
        else:
            raise ValueError("Plane must be one of 'XY', 'XZ', or 'YZ'.")
