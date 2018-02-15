import numpy as np
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
    hdr = np.zeros((3, image.shape[0], image.shape[1]));
    sum = np.zeros((3, image.shape[0], image.shape[1]));
    wij = np.zeros((3, image.shape[0], image.shape[1]));
    m = np.zeros((3, image.shape[0], image.shape[1]));

    for i in range(0, num_exposures):
        print('Adding picture', i + 1, 'of', num_exposures);
        image = cv2.imread(filenames[i]);
        print(filenames[i])

        w = np.array(w)

        np.putmask(wij[2, :, :], wij[2, :, :] == wij[2, :, :], w[255 - image[:, :, 2]])
        np.putmask(wij[1, :, :], wij[1, :, :] == wij[1, :, :], w[255 - image[:, :, 1]])
        np.putmask(wij[0, :, :], wij[0, :, :] == wij[0, :, :], w[255 - image[:, :, 0]])

        sum[2, :, :] = np.add(sum[2, :, :], wij[2, :, :]);
        sum[1, :, :] = np.add(sum[1, :, :], wij[1, :, :]);
        sum[0, :, :] = np.add(sum[0, :, :], wij[0, :, :]);

        np.putmask(m[2, :, :], m[2, :, :] == m[2, :, :], gRed[image[:, :, 2]] - dt[0, i])
        np.putmask(m[1, :, :], m[1, :, :] == m[1, :, :], gGreen[image[:, :, 1]] - dt[0, i])
        np.putmask(m[0, :, :], m[0, :, :] == m[0, :, :], gBlue[image[:, :, 0]] - dt[0, i])

        # If a pixel is saturated, its information and that gathered
        # from all prior pictures with longer exposure times is unreliable.
        # Thus we ignore its influence on the weighted sum(influence of the
        # same pixel from prior pics with longer exposure time ignored as well)
        saturated_pixels = np.ones((3, image.shape[0], image.shape[1]));

        [saturated_pixels] = markSaturatedPixels(saturated_pixels, image[:, :, 2]);
        [saturated_pixels] = markSaturatedPixels(saturated_pixels, image[:, :, 1]);
        [saturated_pixels] = markSaturatedPixels(saturated_pixels, image[:, :, 0]);

        # add the weighted sum of the current pic to the resulting hdr radiance map
        hdr = hdr + np.multiply(wij, m);

        # remove saturated pixels from the radiance map and the
        # sum(saturated pixels are zero in the saturated_pixels matrix, all others are one)
        hdr = np.multiply(hdr, saturated_pixels);
        sum = np.multiply(sum, saturated_pixels);

    # For those pixels that even in the picture
    # with the smallest exposure time still are
    # saturated we approximate the radiance only from that picture
    # instead of taking the weighted sum
    hdr = np.where((hdr == 0), m, hdr)

    # Don't multiply with the weights since they are zero for saturated
    # pixels.m contains the logRadiance value from the last pic, that
    # one with the longest exposure time.

    # Fix the sum for those pixels to avoid division by zero
    np.putmask(sum, sum == 0, 1);

    # normalize
    hdr = hdr / sum;
    hdr = np.exp(hdr);
    return [hdr];
