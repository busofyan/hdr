import numpy as N
import math
import cv2

from sample import sample


def make_image_matrix(dir_name, filenames, num_pixels):
    # number of exposures
    num_exposures = filenames.shape[0]

    # Create the vector of sample indices
    # We need N(P-1) > (Zmax - Zmin)
    # Assuming the maximum (Zmax - Zmin) = 255,
    # N = (255 * 2) / (P-1) clearly fulfills this requirement
    num_samples = int(N.ceil(255 * 2 / (num_exposures - 1)) * 2)

    # create a random sampling matrix, telling us which
    # pixels of the original image we want to sample
    # using ceil fits the indices into the range [1,numPixels+1],
    # i.e. exactly the range of indices of zInput
    step = num_pixels / num_samples
    sample_indices = N.floor(N.array(N.arange(1., num_pixels, step)))
    sample_indices = N.transpose(sample_indices)

    # allocate resulting matrices
    z_red = N.zeros((num_samples, num_exposures))
    z_green = N.zeros((num_samples, num_exposures))
    z_blue = N.zeros((num_samples, num_exposures))

    for i in range(0, num_exposures):
        # read the nth image
        image = cv2.imread(dir_name + filenames[i])

        # sample the image for each color channel
        [z_red_temp, z_green_temp, z_blue_temp] = sample(image, sample_indices)

        # build the resulting, small image consisting
        # of samples of the original image
        z_red[:, i] = z_red_temp
        z_green[:, i] = z_green_temp
        z_blue[:, i] = z_blue_temp

    return [z_red, z_green, z_blue, sample_indices]
