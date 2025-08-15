# -*- coding: utf-8 -*-
"""
Module: slice_loader.py
Author: Gemini
Description: Discovers, filters, and sorts 2D slice image files from a directory.
             This module is designed to be the first step in the mSLA processing
             pipeline, providing a clean, ordered list of slice files for the
             voxel engine.
"""

import re
import os
from pathlib import Path
from typing import List, Optional, Tuple

class SliceLoader:
    """
    Scans a directory for slice files, filters them, and provides a
    numerically sorted list of their paths.

    This class correctly handles:
    - Prefixed and padded filenames (e.g., 'slice_0001.png').
    - Naturally numbered filenames (e.g., '1.png', '10.png').
    - Ignoring common non-slice image files and other data files.
    """
    # Supported image file extensions for slices.
    SUPPORTED_EXTENSIONS = {'.png', '.tiff', '.tif', '.bmp'}

    # Common metadata/preview filenames to explicitly ignore.
    IGNORED_FILENAMES = {'preview.png', '3d.png', 'thumbnail.png'}

    def __init__(self, directory: str):
        """
        Initializes the SliceLoader and scans the specified directory.

        Args:
            directory (str): The path to the directory containing slice files.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        self.directory = Path(directory)
        self.slice_files: List[Path] = []

        if not self.directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        self._find_and_sort_slices()

    def _extract_number(self, path: Path) -> Optional[int]:
        """
        Extracts the first integer sequence from a filename stem.

        Args:
            path (Path): The file path to process.

        Returns:
            Optional[int]: The extracted number as an integer, or None if no
                           number is found.
        """
        # Search for one or more digits (\d+) in the filename's stem (no extension).
        match = re.search(r'(\d+)', path.stem)
        if match:
            return int(match.group(1))
        return None

    def _find_and_sort_slices(self):
        """
        Internal method to perform the core logic of finding, filtering,
        and sorting the slice files in the target directory.
        """
        print(f"Scanning directory: {self.directory}")
        
        candidates: List[Tuple[int, Path]] = []

        for f in self.directory.iterdir():
            # Rule 1: Must be a file.
            if not f.is_file():
                continue

            # Rule 2: Must have a supported image extension.
            if f.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            # Rule 3: Must not be in the explicit ignore list.
            if f.name.lower() in self.IGNORED_FILENAMES:
                continue

            # Rule 4: Must contain a number for sorting.
            num = self._extract_number(f)
            if num is not None:
                candidates.append((num, f))

        # Sort candidates based on the extracted number (the first item in the tuple).
        # This achieves "natural sorting" (e.g., 2 comes before 10).
        candidates.sort(key=lambda x: x[0])

        # The final list contains only the sorted Path objects.
        self.slice_files = [path for num, path in candidates]
        
        print(f"Found and sorted {len(self.slice_files)} slice files.")

    def get_slice_list(self) -> List[Path]:
        """
        Returns the complete, sorted list of slice file paths.

        Returns:
            List[Path]: A list of Path objects for each valid slice file.
        """
        return self.slice_files

    def __len__(self) -> int:
        """Returns the number of slices found."""
        return len(self.slice_files)

    def __getitem__(self, index: int) -> Path:
        """Allows accessing slices by index."""
        return self.slice_files[index]


# --- Example Usage ---
if __name__ == '__main__':
    # To test this module, create a temporary directory and some dummy files.
    print("--- SliceLoader Test ---")
    
    # Create a test directory structure.
    test_dir = Path("./temp_slice_test_dir")
    if not test_dir.exists():
        test_dir.mkdir()

    # Create dummy files to simulate a real slice folder.
    dummy_files = [
        "slice_0001.png", "slice_0002.png", "slice_0010.png", "slice_0009.png",
        "1.png", "2.png", "10.png", "9.png",
        "preview.png", "config.ini", "model.gcode", "3d.png", "archive.zip",
        "data_with_number_123.txt"
    ]
    
    print(f"\nCreating dummy files in: {test_dir.resolve()}")
    for fname in dummy_files:
        (test_dir / fname).touch()

    try:
        # Instantiate the loader.
        loader = SliceLoader(str(test_dir))

        # Get the sorted list of files.
        sorted_slices = loader.get_slice_list()

        if sorted_slices:
            print("\nSorted slice files found:")
            for i, slice_path in enumerate(sorted_slices):
                # Print just the filename for clarity.
                print(f"  {i:02d}: {slice_path.name}")
        else:
            print("\nNo valid slice files were found.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    finally:
        # Clean up the dummy files and directory.
        print(f"\nCleaning up test directory...")
        for f in test_dir.iterdir():
            f.unlink()
        test_dir.rmdir()
        print("Cleanup complete.")

