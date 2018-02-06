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
        w = N.array(w)

        N.putmask(wij[2, :, :], wij[2, :, :] > -1, w[255 - image[:, :, 2]])
        N.putmask(wij[1, :, :], wij[1, :, :] > -1, w[255 - image[:, :, 1]])
        N.putmask(wij[0, :, :], wij[0, :, :] > -1, w[255 - image[:, :, 0]])

        sum[2, :, :] = N.add(sum[2, :, :], wij[2, :, :]);
        sum[1, :, :] = N.add(sum[1, :, :], wij[1, :, :]);
        sum[0, :, :] = N.add(sum[0, :, :], wij[0, :, :]);

        N.putmask(m[2, :, :], m[2, :, :] > -1, gRed[image[:, :, 2]] - dt[0, i])
        N.putmask(m[1, :, :], m[1, :, :] > -1, gRed[image[:, :, 1]] - dt[0, i])
        N.putmask(m[0, :, :], m[0, :, :] > -1, gRed[image[:, :, 0]] - dt[0, i])

        # If a pixel is saturated, its information and that gathered
        # from all prior pictures with longer exposure times is unreliable.
        # Thus we ignore its influence on the weighted sum(influence of the
        # same pixel from prior pics with longer exposure time ignored as well)
        saturatedPixels = N.ones((3, image.shape[0], image.shape[1]));

        [saturatedPixels] = markSaturatedPixels(saturatedPixels, image[:, :, 2]);
        [saturatedPixels] = markSaturatedPixels(saturatedPixels, image[:, :, 1]);
        [saturatedPixels] = markSaturatedPixels(saturatedPixels, image[:, :, 0]);
        print(saturatedPixels)

        # Mark the saturated pixels from a certain channel in * all three * channels
        dim = image.shape[0] * image.shape[1];

        # add the weighted sum of the current pic to the resulting hdr radiance map
        hdr = hdr + N.multiply(wij, m);

        # remove saturated pixels from the radiance map and the
        # sum(saturated pixels are zero in the saturatedPixels matrix, all others are one)
        hdr = N.multiply(hdr, saturatedPixels);
        sum = N.multiply(sum, saturatedPixels);

    # For those pixels that even in the picture
    # with the smallest exposure time still are
    # saturated we approximate the radiance only from that picture
    # instead of taking the weighted sum
    saturatedPixelIndices = N.where(hdr == 0);
    #print(saturatedPixelIndices);

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
