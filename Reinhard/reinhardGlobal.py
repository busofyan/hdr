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
import makeLuminanceMap

def reinhardGlobal( hdr, a, saturation):

    print('Computing luminance map\n');
    luminanceMap = makeLuminanceMap(hdr);

    numPixels = N.size(hdr,1) * N.size(hdr,2);

    # small delta to avoid taking log(0) when encountering black pixels in the
    # luminance map
    delta = 0.0001;

    # compute the key of the image, a measure of the
    # average logarithmic luminance, i.e. the subjective brightness of the image a human
    # would approximateley perceive
    key = N.exp((1 / numPixels) * (N.sum(N.sum(N.log(luminanceMap + delta)))));


    # scale to desired brightness level as defined by the user
    scaledLuminance = luminanceMap * (a/key);

    # all values are now mapped to the range [0,1]
    ldrLuminanceMap = N.divide(scaledLuminance, (scaledLuminance + 1));


    ldrPic = N.zeros(N.size(hdr));

    # re-apply color according to Fattals paper "Gradient Domain High Dynamic
    # Range Compression"
    for i in range(1, 3):
        # (hdr(:,:,i) ./ luminance) MUST be between 0 an 1!!!!
        # ...but hdr often contains bigger values than luminance!!!???
        # so the resulting ldr pic needs to be clamped
        ldrPic[:,:,i] = N.pow(N.multiply((N.divide((hdr[:,:,i], luminanceMap)), saturation)), ldrLuminanceMap);

    # clamp ldrPic to 1
    indices = N.where(ldrPic > 1);
    ldrPic[indices] = 1;

    return [ ldrPic, ldrLuminanceMap ];

