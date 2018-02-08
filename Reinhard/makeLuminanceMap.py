import numpy as np


def makeLuminanceMap(image):
    # compute the hdr luminance map from the hdr radiance map. It is needed as
    # an input for the Reinhard tonemapping operators.
    luminance_map = np.zeros((image.shape[1], image.shape[2]));
    temp_red = image[2, :, :] * 0.2125
    temp_green = image[1, :, :] * 0.7154
    temp_blue = image[0, :, :] * 0.0721
    np.putmask(luminance_map, luminance_map == luminance_map, (temp_red + temp_green + temp_blue))

    return [luminance_map];
