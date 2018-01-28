import numpy as N
from readDir import read_dir
import cv2


# Generates a hdr radiance map from a set of pictures parameters: filenames: a list of filenames
# containing the differently exposed pictures used to make a hdr from
# gRed: camera response function for the red color channel
# gGreen: camera response function for the green color channel
# gBlue: camera response function for the blue color channel

def hdr(filenames, gRed, gGreen, gBlue, w, dt):

    num_exposures = filenames.shape[0]

    # read the first image to get the width and height information
    dir_name = '../images/'
    [filenames, exposures, numExposures] = read_dir(dir_name)

    print('Opening Test Images\n');
    image = cv2.imread(dir_name + filenames[0]);

    # pre - allocate resulting hdr image
    hdr = N.zeros(image.shape[0]);
    sum = N.zeros(image.shape[0]);
    m = [];

    for i in range(0, num_exposures):

        print('Adding picture %i of %i \n', i, num_exposures);
        image = cv2.imread(dir_name + filenames[i]);

        print(w);
        wij = w[image + 1];
        sum = sum + wij;

        m[:, :, 2] = (gRed(image[:, :, 2] + 1) - dt(1, i));
        m[:, :, 1] = (gGreen(image[:, :, 1] + 1) - dt(1, i));
        m[:, :, 0] = (gBlue(image[:, :, 0] + 1) - dt(1, i));

        # If a pixel is saturated, its information and that gathered
        # from all prior pictures with longer exposure times is unreliable.
        # Thus we ignore its influence on the weighted sum(influence of the
        # same pixel from prior pics with longer exposure time ignored as well)

        saturatedPixels = N.ones(N.size(image));

        saturatedPixelsRed = N.where(image[:, :, 2] == 255);
        saturatedPixelsGreen = N.where(image[:, :, 1] == 255);
        saturatedPixelsBlue = N.where(image[:, :, 0] == 255);

        # Mark the saturated pixels from a certain channel in * all three * channels
        dim = N.size(image, 1) * N.size(image, 2);

        saturatedPixels[saturatedPixelsRed] = 0;
        saturatedPixels[saturatedPixelsRed + dim] = 0;
        saturatedPixels[saturatedPixelsRed + 2 * dim] = 0;

        saturatedPixels[saturatedPixelsGreen] = 0;
        saturatedPixels[saturatedPixelsGreen + dim] = 0;
        saturatedPixels[saturatedPixelsGreen + 2 * dim] = 0;

        saturatedPixels[saturatedPixelsBlue] = 0;
        saturatedPixels[saturatedPixelsBlue + dim] = 0;
        saturatedPixels[saturatedPixelsBlue + 2 * dim] = 0;

        # add the weighted sum of the current pic to the resulting hdr radiance map
        hdr = hdr + N.multiply(wij, m);
        # BIN MIR NICHT SICHER MATLAB SCHREIBWEISE .*

        # remove saturated pixels from the radiance map and the
        # sum(saturated pixels are zero in the saturatedPixels matrix, all others are one)
        hdr = N.mutiply(hdr, saturatedPixels);
        sum = N.mutiply(sum, saturatedPixels);
        # BIN MIR NICHT SICHER MATLAB SCHREIBWEISE .*

        # For those pixels that even in the picture
        # with the smallest exposure time still are
        # saturated we approximate the radiance only from that picture

        # instead of taking the weighted sum
        saturatedPixelIndices = N.where(hdr == 0);

    # Don't multiply with the weights since they are zero for saturated
    # pixels.m contains the logRadiance value from the last pic, that
    # one with the longest exposure time.
    hdr[saturatedPixelIndices] = m[saturatedPixelIndices];

    # Fix the sum for those pixels to avoid division by zero
    sum[saturatedPixelIndices] = 1;

    # normalize
    hdr = N.divide(hdr, sum);
    # AUCH WIEDER ./
    hdr = N.exp(hdr);

    return [hdr];
