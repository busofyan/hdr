import numpy as N
import math
import cv2

def makeLuminanceMap(image):
    # Creates a luminance map from an image
    #  The input image is expected to be a 3d matrix of size rows*columns*3
    luminanceMap = N.zeros((image.shape[1], image.shape[2]));
    temp_red = image[2, :, :] * 0.2125
    temp_green = image[1, :, :] * 0.7154
    temp_blue = image[0, :, :] * 0.0721
    N.putmask(luminanceMap, luminanceMap < 257, temp_red + temp_green + temp_blue)

    return [luminanceMap];
