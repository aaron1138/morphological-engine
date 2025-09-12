from PIL import Image
import numpy as np

def create_test_image():
    width, height = 256, 256
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            array[y, x] = [x, y, 128]

    image = Image.fromarray(array, 'RGB')
    image.save("test_assets/input.png")
    print("Created test_assets/input.png")

if __name__ == "__main__":
    create_test_image()
