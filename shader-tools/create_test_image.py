from PIL import Image
import numpy as np

def create_test_image(width, height, filename):
    """Creates a simple colorful test image."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            array[y, x, 0] = x % 256
            array[y, x, 1] = y % 256
            array[y, x, 2] = (x + y) % 256

    img = Image.fromarray(array, 'RGB')
    img.save(filename, 'PNG')
    print(f"Created test image: {filename}")

if __name__ == "__main__":
    create_test_image(256, 256, "shader-tools/test_image.png")
