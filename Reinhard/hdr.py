import numpy as N
from markSaturatedPixels import markSaturatedPixels
import cv2


# Generates a hdr radiance map from a set of pictures parameters: filenames: a list of filenames
# containing the differently exposed pictures used to make a hdr from
# gRed: camera response function for the red color channel
# gGreen: camera response function for the green color channel
# gBlue: camera response function for the blue color channel

def hdr(filenames, gRed, gGreen, gBlue, w, dt):

    num_exposures = filenames.shape[0]

    print('Opening Test Images\n');
    image = cv2.imread(filenames[0]);

    # pre - allocate resulting hdr image
    hdr = N.zeros((3, image.shape[0], image.shape[1]));
    sum = N.zeros((3, image.shape[0], image.shape[1]));
    wij = N.zeros((3, image.shape[0], image.shape[1]));
    m = N.zeros((3, image.shape[0], image.shape[1]));

    for i in range(0, num_exposures):
        print('Adding picture', i ,'of', num_exposures);
        image = cv2.imread(filenames[i]);
        print(filenames[i])

        for a in range(0, (image.shape[0])):
            for b in range(0, (image.shape[1])):
                wij[2, a, b] = w[255 - image[:, :, 2][a, b]];
                wij[1, a, b] = w[255 - image[:, :, 1][a, b]];
                wij[0, a, b] = w[255 - image[:, :, 0][a, b]];

        sum[:, :, 2] = N.add(sum[:, :, 2], wij[:, :, 2]);
        sum[:, :, 1] = N.add(sum[:, :, 1], wij[:, :, 1]);
        sum[:, :, 0] = N.add(sum[:, :, 0], wij[:, :, 0]);

        for a in range(0, (image.shape[0])):
            for b in range(0, (image.shape[1])):
                m[2, a, b] = gRed[image[:, :, 2][a, b]] - dt[0, i];
                m[1, a, b] = gGreen[image[:, :, 1][a, b]] - dt[0, i];
                m[0, a, b] = gBlue[image[:, :, 0][a, b]] - dt[0, i];

        # If a pixel is saturated, its information and that gathered
        # from all prior pictures with longer exposure times is unreliable.
        # Thus we ignore its influence on the weighted sum(influence of the
        # same pixel from prior pics with longer exposure time ignored as well)

        saturatedPixelsRedTemp = [];
        saturatedPixelsGreenTemp = [];
        saturatedPixelsBlueTemp = [];
        saturatedPixelsRed = [];
        saturatedPixelsGreen = [];
        saturatedPixelsBlue = [];

        saturatedPixels = N.ones((3, image.shape[0], image.shape[1]));

        for i in range(0, image.shape[1]):
            for j in range(0, image.shape[0]):
                saturatedPixelsRedTemp.append(image[:, :, 2][j, i]);
                saturatedPixelsGreenTemp.append(image[:, :, 1][j, i]);
                saturatedPixelsBlueTemp.append(image[:, :, 0][j, i]);

        for i in range(0, (saturatedPixelsRedTemp.__len__())):
            if saturatedPixelsRedTemp[i] == 255:
                saturatedPixelsRed.append(i);

        for i in range(0, (saturatedPixelsGreenTemp.__len__())):
            if saturatedPixelsGreenTemp[i] == 255:
                saturatedPixelsGreen.append(i);

        for i in range(0, (saturatedPixelsBlueTemp.__len__())):
            if saturatedPixelsBlueTemp[i] == 255:
                saturatedPixelsBlue.append(i);

        # Mark the saturated pixels from a certain channel in * all three * channels
        dim = image.shape[0] * image.shape[1];

        [saturatedPixels] = markSaturatedPixels(saturatedPixels, saturatedPixelsRed);
        [saturatedPixels] = markSaturatedPixels(saturatedPixels, saturatedPixelsGreen);
        [saturatedPixels] = markSaturatedPixels(saturatedPixels, saturatedPixelsBlue);

        # add the weighted sum of the current pic to the resulting hdr radiance map
        hdr = hdr + N.multiply(wij, m);
        # BIN MIR NICHT SICHER MATLAB SCHREIBWEISE .*

        # remove saturated pixels from the radiance map and the
        # sum(saturated pixels are zero in the saturatedPixels matrix, all others are one)
        #hdr = N.mutiply(hdr, saturatedPixels);
        #sum = N.mutiply(sum, saturatedPixels);
        # BIN MIR NICHT SICHER MATLAB SCHREIBWEISE .*

        # For those pixels that even in the picture
        # with the smallest exposure time still are
        # saturated we approximate the radiance only from that picture

        # instead of taking the weighted sum
        #saturatedPixelIndices = N.where(hdr == 0);

    # Don't multiply with the weights since they are zero for saturated
    # pixels.m contains the logRadiance value from the last pic, that
    # one with the longest exposure time.
    #hdr[saturatedPixelIndices] = m[saturatedPixelIndices];

    # Fix the sum for those pixels to avoid division by zero
    #sum[saturatedPixelIndices] = 1;

    # normalize
    #hdr = N.divide(hdr, sum);
    # AUCH WIEDER ./
    #hdr = N.exp(hdr);

    return [hdr];
