from PIL import Image
import numpy as np

width, height = 256, 256
array = np.zeros((height, width, 3), dtype=np.uint8)
array[:, :, 0] = np.linspace(0, 255, width)
array[:, :, 1] = np.linspace(0, 255, height).reshape(-1, 1)
array[:, :, 2] = (np.linspace(0, 255, width) + np.linspace(0, 255, height).reshape(-1, 1)) % 256

img = Image.fromarray(array)
img.save("test_image.png")
