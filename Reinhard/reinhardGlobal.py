# Implements the Reinhard global tonemapping operator.
#
# parameters:
# hdr: a rows * cols * 3 matrix representing a hdr radiance map
#
# a: defines the desired brightness level of the resulting tonemapped
# picture. Lower values generate darker pictures, higher values brighter
# pictures. Use values like 0.18, 0.36 or 0.72 for bright pictures, 0.09,
# 0.045, ... for darker pictures.
#
# saturation: a value between 0 and 1 defining the desired saturation of
# the resulting tonemapped image

import numpy as np
from makeLuminanceMap import makeLuminanceMap


def reinhardGlobal(hdr, a, saturation):
    print('Computing luminance map\n');
    [luminanceMap] = makeLuminanceMap(hdr);

    num_pixels = (hdr.shape[1] * hdr.shape[2])
    # small delta to avoid taking log(0) when encountering black pixels in the
    # luminance map
    delta = 0.0001;

    # compute the key of the image, a measure of the
    # average logarithmic luminance, i.e. the subjective brightness of the image a human
    # would approximateley perceive
    key = (np.exp((1.0 / num_pixels) * np.sum(np.sum(np.log(np.add(luminanceMap, delta))))));

    # scale to desired brightness level as defined by the user
    scaledLuminance = np.multiply(luminanceMap, a / key)

    # all values are now mapped to the range [0,1]
    ldrLuminanceMap = np.divide(scaledLuminance, (scaledLuminance + 1));

    # re-apply color according to Fattals paper "Gradient Domain High Dynamic
    # Range Compression"
    ldrPic = np.multiply((np.power((hdr / luminanceMap), saturation)), ldrLuminanceMap)

    # clamp ldrPic to 1
    np.putmask(ldrPic, ldrPic > 1, 1)

    # convert color values to RGB
    writeGlobal = np.ceil(ldrPic * 255)

    # stack every color matrix (axis) to build the image
    writeGlobal = np.stack((writeGlobal[0, :, :], writeGlobal[1, :, :], writeGlobal[2, :, :]), axis=-1);
    showGlobal = np.stack((ldrPic[0, :, :], ldrPic[1, :, :], ldrPic[2, :, :]), axis=-1);

    return [showGlobal, writeGlobal];
