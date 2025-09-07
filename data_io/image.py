import numpy as np
from PIL import Image
from pathlib import Path
from typing import List

def read_image(filepath: Path) -> np.ndarray:
    """
    Reads an image file and returns it as a grayscale NumPy array.

    Args:
        filepath (Path): The path to the image file.

    Returns:
        np.ndarray: A 2D NumPy array representing the grayscale image.
    """
    try:
        with Image.open(filepath) as img:
            # Convert to grayscale ('L' mode for 8-bit pixels)
            grayscale_img = img.convert('L')
            return np.array(grayscale_img)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found at: {filepath}")
    except Exception as e:
        raise IOError(f"Error reading image file {filepath}: {e}")

def write_image(filepath: Path, data: np.ndarray):
    """
    Writes a NumPy array to a PNG image file.

    Args:
        filepath (Path): The path to save the image file to.
        data (np.ndarray): The 2D NumPy array to save.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")

    try:
        # Ensure the parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create an image from the array. The mode is inferred from the array's dtype.
        img = Image.fromarray(data)
        img.save(filepath, 'PNG')
    except Exception as e:
        raise IOError(f"Error writing image file {filepath}: {e}")

# --- Example Usage ---
if __name__ == '__main__':
    print("--- I/O Image Module Test ---")

    test_dir = Path("./temp_io_image_test_dir")
    test_dir.mkdir(exist_ok=True)
    test_filepath = test_dir / "test_image.png"

    try:
        # Create a dummy image (NumPy array) with a gradient
        original_data = np.zeros((100, 200), dtype=np.uint8)
        original_data[:, :] = np.linspace(0, 255, 200, dtype=np.uint8)

        print(f"Created original data with shape: {original_data.shape}")

        # 1. Test write operation
        write_image(test_filepath, original_data)
        print(f"Successfully wrote image to: {test_filepath}")
        assert test_filepath.exists()

        # 2. Test read operation
        loaded_data = read_image(test_filepath)
        print(f"Successfully read image from: {test_filepath}")
        print(f"Loaded data with shape: {loaded_data.shape}")

        # 3. Verification
        if np.array_equal(original_data, loaded_data):
            print("\nVerification SUCCESS: Original and loaded data are identical.")
        else:
            # Provide more details on failure
            diff = np.abs(original_data.astype(float) - loaded_data.astype(float))
            print(f"\nVerification FAILED: Original and loaded data differ.")
            print(f"  - Max difference: {np.max(diff)}")
            print(f"  - Mismatched pixels: {np.sum(diff > 0)}")
            raise RuntimeError("I/O verification failed.")

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
    finally:
        # Clean up the test file and directory
        if test_filepath.exists():
            test_filepath.unlink()
        if test_dir.exists():
            test_dir.rmdir()
        print("\nTest cleanup complete.")
