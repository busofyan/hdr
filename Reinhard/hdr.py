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

    print('Opening Test Images\n');
    image = cv2.imread(filenames[0]);

    # pre - allocate resulting hdr image
    hdr = N.zeros((image.shape[0],image.shape[1]));
    sum_red = N.zeros((image.shape[0],image.shape[1]));
    sum_green = N.zeros((image.shape[0], image.shape[1]));
    sum_blue = N.zeros((image.shape[0], image.shape[1]));
    wij_red = N.zeros((image.shape[0],image.shape[1]));
    wij_green = N.zeros((image.shape[0], image.shape[1]));
    wij_blue = N.zeros((image.shape[0], image.shape[1]));

    m_red = N.zeros((image.shape[0], image.shape[1]));
    m_green = N.zeros((image.shape[0], image.shape[1]));
    m_blue = N.zeros((image.shape[0], image.shape[1]));

    for i in range(0, num_exposures):
        print('Adding picture', i ,'of', num_exposures);
        image = cv2.imread(filenames[i]);
        print(filenames[i])

        red_channel = image[:, :, 2];
        green_channel = image[:, :, 1];
        blue_channel = image[:, :, 0];

        for a in range(0, image.shape[0]):
            for b in range(0, image.shape[1]):
                wij_red[a,b] = w[255-red_channel[a, b]];
                wij_green[a, b] = w[255 - green_channel[a, b]];
                wij_blue[a, b] = w[255 - blue_channel[a, b]];

        sum_red = N.add(sum_red , wij_red);
        sum_green = N.add(sum_green, wij_green);
        sum_blue = N.add(sum_blue, wij_blue);


        m_red = [gRed[red_channel] - dt[0, i]];
        m_green = (gGreen[green_channel + 1] - dt[0, i]);
        m_blue = (gBlue[blue_channel + 1] - dt[0, i]);

        print(m_red);

        # If a pixel is saturated, its information and that gathered
        # from all prior pictures with longer exposure times is unreliable.
        # Thus we ignore its influence on the weighted sum(influence of the
        # same pixel from prior pics with longer exposure time ignored as well)

        saturatedPixels = N.ones((image.shape[0],image.shape[1]));

        saturatedPixelsRed = N.where(image[:, :, 2] == 255);
        saturatedPixelsGreen = N.where(image[:, :, 1] == 255);
        saturatedPixelsBlue = N.where(image[:, :, 0] == 255);

        # Mark the saturated pixels from a certain channel in * all three * channels
        dim = image.shape[0] * image.shape[1];

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
