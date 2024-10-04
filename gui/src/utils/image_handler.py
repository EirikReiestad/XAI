from PIL import Image
import os


def open_image(path: str):
    if os.path.exists(path):
        image = Image.open(path)
    else:
        image = Image.new("RGB", (1, 1), (255, 255, 255))
    return image
