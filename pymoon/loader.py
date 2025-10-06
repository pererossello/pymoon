import numpy as np
from PIL import Image


def resize_image(img, new_shape):
    """
    Resize a numpy RGB image to new_shape.
    """
    img_resized = img.resize((new_shape[1], new_shape[0]), Image.BILINEAR)
    return img_resized

