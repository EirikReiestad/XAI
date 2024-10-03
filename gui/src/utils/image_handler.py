from PIL import Image


def open_image(path: str):
    image = Image.open(path)
    return image
