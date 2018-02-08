import numpy as np
import cv2
from sample import sample


def make_image_matrix(dir_name, filenames, num_pixels):
    # number of exposures
    num_exposures = filenames.shape[0]

    # Create the vector of sample indices
    # We need np(P-1) > (Zmax - Zmin)
    # Assuming the maximum (Zmax - Zmin) = 255,
    # np = (255 * 2) / (P-1) clearly fulfills this requirement
    num_samples = int(np.ceil(255.0 * 2.0 / (num_exposures - 1.0)) * 2.0)

    # create a random sampling matrix, telling us which
    # pixels of the original image we want to sample
    # using ceil fits the indices into the range [1,numPixels+1],
    # i.e. exactly the range of indices of zInput
    step = np.float128(num_pixels) / np.float128(num_samples)

    sample_indices = np.floor(np.array(np.arange(0.0, np.float128(num_pixels), step)))
    sample_indices = sample_indices.astype(int);

    # allocate resulting matrices
    z_red = np.zeros((num_samples, num_exposures))
    z_green = np.zeros((num_samples, num_exposures))
    z_blue = np.zeros((num_samples, num_exposures))

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
