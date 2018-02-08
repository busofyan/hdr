import numpy as np


def markSaturatedPixels(saturated_pixels, image):
    image = np.where(image == 255, 0, 1)
    saturated_pixels = np.where((image + saturated_pixels < 2), 0, 1)

    return [saturated_pixels];
