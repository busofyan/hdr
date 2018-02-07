import cv2
import numpy as N
import math

from readDir import read_dir
from weight import weight
from makeImageMatrix import make_image_matrix
from gsolve import gsolve
from hdr import hdr
from reinhardGlobal import reinhardGlobal
#from reinhardLocal import reinhardLocal


dir_name = '../images_130/'
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

B = N.zeros((N.size(z_red), numExposures));

print('Creating exposures matrix B\n');
for i in range(0, numExposures):
    B[:,i] = math.log(exposures[i]);

# % solve the system for each color channel
print('Solving for red channel\n');
[gRed, lERed] = gsolve(z_red, B, l, weights);

print('Solving for green channel\n')
[gGreen, lEGreen] = gsolve(z_green, B, l, weights);

print('Solving for blue channel\n')
[gBlue, lEBlue] = gsolve(z_blue, B, l, weights);

# compute the hdr radiance map
print('Computing hdr image\n')
[hdrMap] = hdr(filenames, gRed, gGreen, gBlue, weights, B);

# compute the hdr luminance map from the hdr radiance map. It is needed as
# an input for the Reinhard tonemapping operators.
print('Computing luminance map\n');
luminance = N.zeros((tmp.shape[0], tmp.shape[1]));

temp_red = hdrMap[2, :, :] * 0.2125
temp_green = hdrMap[1, :, :] * 0.7154
temp_blue = hdrMap[0, :, :] * 0.0721
N.putmask(luminance, luminance < 257, temp_red + temp_green + temp_blue)

# apply Reinhard local tonemapping operator to the hdr radiance map
print('Tonemapping - Reinhard local operator\n');
saturation = 0.6;
eps = 0.05;
phi = 8;
#[ldrLocal, luminanceLocal, v, v1Final, sm ]  = reinhardLocal(hdrMap, saturation, eps, phi);

# apply Reinhard global tonemapping oparator to the hdr radiance map
print('Tonemapping - Reinhard global operator\n');
# specify resulting brightness of the tonampped image. See reinhardGlobal.m
# for details
a = 0.72;

# specify saturation of the resulting tonemapped image. See reinhardGlobal.m
# for details
saturation = 0.6;
[ldrGlobal, luminanceGlobal] = reinhardGlobal(hdrMap, a, saturation);
print(ldrGlobal)
#create a window for display.
#cv2.namedWindow("Display window", 12345);
#cv2.imshow("reinhardGlobal.jpg", ldrGlobal);
#cv2.waitKey(0)

print('Finished!\n');

