import cv2
import numpy as N

from readDir import read_dir
from weight import weight
from makeImageMatrix import make_image_matrix

dir_name = '../images/'
[filenames, exposures, numExposures] = read_dir(dir_name)

print('Opening Test Images\n')
tmp = cv2.imread(dir_name + filenames[0])

num_pixels = tmp.shape[0] * tmp.shape[1]
num_exposurs = filenames.shape[0]

## lamda smoothing factor
l = 50

print('Computing weighting function\n')
# precompute the weighing function value
# for each pixel
#
weights = []
#
for i in range(1, 257):
    weights.append(weight(i, 1, 256))

# load and sample the images
[z_red, z_green, z_blue, sampleIndices] = make_image_matrix(dir_name, filenames, num_pixels)

#
## tmp = [cv2.imread(dir_name + fn) for fn in filenames]
