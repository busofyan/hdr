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
#
import numpy as N
from makeLuminanceMap import makeLuminanceMap


def reinhardGlobal(hdr, a, saturation):
    print('Computing luminance map\n');
    luminanceMap = makeLuminanceMap(hdr);

    num_pixels = (hdr.shape[1] * hdr.shape[2])
    # small delta to avoid taking log(0) when encountering black pixels in the
    # luminance map
    delta = 0.0001;

    # compute the key of the image, a measure of the
    # average logarithmic luminance, i.e. the subjective brightness of the image a human
    # would approximateley perceive
    key = (N.exp((1.0/num_pixels) * N.sum(N.sum(N.log(N.add(luminanceMap, delta))))));

    # scale to desired brightness level as defined by the user
    scaledLuminance = N.multiply(luminanceMap, a/key)

    # all values are now mapped to the range [0,1]
    ldrLuminanceMap = N.divide(scaledLuminance, (scaledLuminance + 1));

    ldrPic = N.zeros((3, hdr.shape[1], hdr.shape[2]));

    # re-apply color according to Fattals paper "Gradient Domain High Dynamic
    # Range Compression"
    ldrPic = N.multiply((N.power((hdr / luminanceMap), saturation)), ldrLuminanceMap)

    # clamp ldrPic to 1
    N.putmask(ldrPic, ldrPic > 1, 1)

    # convert color values to RGB
    #ldrPic = N.ceil(ldrPic * 255)

    #for i in range(0, hdr.shape[1]):
    #    for i in range(0, hdr.shape[1]):

    img = N.zeros((hdr.shape[1], hdr.shape[2], 3));
    k = 0
    r = 0
    for i in range(0,hdr.shape[1]):
        r = 0
        for j in range(0, hdr.shape[2]):
            axis = 0
            img[k, r, axis] = ldrPic[0, :, :][k, r]
            axis += 1
            img[k, r, axis] = ldrPic[1, :, :][k, r]
            axis += 1
            img[k, r, axis] = ldrPic[2, :, :][k, r]
            r += 1
        k += 1

    return [img, ldrLuminanceMap];
