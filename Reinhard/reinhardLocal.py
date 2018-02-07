# Implements the Reinhard local tonemapping operator
#
# parameters:
# hdr: high dynamic range radiance map, a matrix of N.size rows * columns * 3
# luminance map: the corresponding lumiance map of the hdr image
#
import numpy as N
import cv2
from makeLuminanceMap import makeLuminanceMap

def reinhardLocal( hdr, saturation, eps, phi ):
    print('Computing luminance map\n');
    luminanceMap = makeLuminanceMap(hdr);
    key = 0.18;
    alpha = 1 / (2 * N.sqrt(2));
    v1 = N.zeros((9, hdr.shape[1], hdr.shape[2]));
    v = N.zeros((8, hdr.shape[1], hdr.shape[2]));

    # compute nine gaussian filtered version of the hdr luminance map, such
    # that we can compute eight differences. Each image gets filtered by a
    # standard gaussian filter, each time with sigma 1.6 times higher than
    # the sigma of the predecessor.
    for scale in range(0, 9):
        s = N.power(1.6,(scale));
        sigma = alpha * s;
        kernelRadius = N.ceil(2 * sigma);
        kernelSize2 = 2 * kernelRadius + 1;
        # dicretize gaussian filter to a fixed N.size kernel.
        # a radius of 2*sigma should keep the error low enough...
        kernelSize = 1;
        luminanceMap = N.array(luminanceMap)

        gaussKernelHorizontal = cv2.getGaussianKernel(int(kernelSize2), sigma, cv2.CV_32F)
        #gausdiemaus = cv2.GaussianBlur(luminanceMap, (int(kernelSize), int(kernelSize)), 0)
        v2 = cv2.GaussianBlur(luminanceMap, (int(kernelSize), int(kernelSize)), 0)
        v1[scale, :, :] = cv2.filter2D(v2, -1, gaussKernelHorizontal)
        print(v1[scale, :, :])
        kernelSize += 2

    for i in range(0, 8):
        v[i, :, :] = N.divide(N.absolute(v1[i, :, :] - v1[i+1, :, :]) , ((N.power(2, phi)) * key / (N.power(s, 2)) + v1[i, :, :]))

    sm = N.zeros((v.shape[1], v.shape[2]));

    for i in range(0, v.shape[1]):
        for j in range(0, v.shape[2]):
            for scale in range(0, v.shape[0]):
                # choose the biggest possible neighbourhood where v(i,j,scale)
                # is still smaller than a certain epsilon.
                # Note that we need to choose that neighbourhood which is
                # as big as possible but all smaller neighbourhoods also
                # fulfill v(i,j,scale) < eps !!!
                if v[scale, i, j] > eps:
                    # if we already have a high contrast change in the
                    # first scale we can only use that one
                    if scale == 1:
                        sm[i, j] = 1;

                    # if we have a contrast change bigger than epsilon, we
                    # know that in scale scale-1 the contrast change was
                    # smaller than epsilon and use that one
                    if scale > 1:
                        sm[i,j] = scale - 1;
                    break;

    # all areas in the pic that have very small variations and therefore in
    # any scale no contrast change > epsilon will not have been found in
    # the loop above.
    # We manually need to assign them the biggest possible scale.

    N.putmask(sm, sm[:,:] == 0, 1);
    sm = N.array(sm)
    v1Final = N.zeros((v.shape[1], v.shape[2]));

    # build the local luminance map with luminance values taken
    # from the neighbourhoods with appropriate scale
    for x in range(0, v.shape[1]):
        for y in range(0, v.shape[2]):
            v1Final[x, y] = v1[int(sm[x,y]), x, y];

    # Do the actual tonemapping
    luminanceCompressed = N.divide(luminanceMap, (1 + v1Final));

    ldrPic = N.zeros((3, hdr.shape[1], hdr.shape[2]));

    # re-apply color according to Fattals paper "Gradient Domain High Dynamic
    # Range Compression"

    for i in range(0, 3):
        # (hdr(:,:,i) ./ luminance) MUST be between 0 an 1!!!!
        # ...but hdr often contains bigger values than luminance!!!???
        # so the resulting ldr pic needs to be clamped
        ldrPic[i, :, :] = N.multiply(N.power(N.divide(hdr[i, :, :], luminanceMap), saturation), luminanceCompressed);

    # clamp ldrPic to 1
    N.putmask(ldrPic, ldrPic > 1, 1)

    # use dirty hack for matrix reshape operation
    img = N.zeros((hdr.shape[1], hdr.shape[2], 3));
    k = 0
    r = 0
    for i in range(0, hdr.shape[1]):
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

    return [img, luminanceCompressed, v, v1Final, sm];