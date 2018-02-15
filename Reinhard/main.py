import cv2
import numpy as np
import math

from readDir import read_dir
from weight import weight
from makeImageMatrix import make_image_matrix
from gsolve import gsolve
from hdr import hdr
from reinhardGlobal import reinhardGlobal
from reinhardLocal import reinhardLocal

dir_name = '../desk/'
[filenames, exposures, numExposures] = read_dir(dir_name)

print('Opening Test Images\n')
tmp = cv2.imread(dir_name + filenames[0])

num_pixels = tmp.shape[0] * tmp.shape[1]

# lamda smoothing factor
l = 50

print('Computing weighting function\n')
# precompute the weighing function value for each pixel
weights = []

for i in range(1, 257):
    weights.append(weight(i, 1, 256))

# load and sample the images
[z_red, z_green, z_blue] = make_image_matrix(dir_name, filenames, num_pixels)

B = np.zeros((np.size(z_red), numExposures));

print('Creating exposures matrix B\n');
for i in range(0, numExposures):
    B[:, i] = math.log(exposures[i]);

# solve the system for each color channel
print('Solving for red channel\n');
[gRed] = gsolve(z_red, B, l, weights);

print('Solving for green channel\n')
[gGreen] = gsolve(z_green, B, l, weights);

print('Solving for blue channel\n')
[gBlue] = gsolve(z_blue, B, l, weights);

# compute the hdr radiance map
print('Computing hdr image\n')
[hdrMap] = hdr(filenames, gRed, gGreen, gBlue, weights, B);

# apply Reinhard local tonemapping operator to the hdr radiance map
print('Tonemapping - Reinhard local operator\n');
saturation = 0.6;
eps = 0.05;
phi = 8;
[showLocal, writeLocal] = reinhardLocal(hdrMap, saturation, eps, phi);

# apply Reinhard global tonemapping oparator to the hdr radiance map
print('Tonemapping - Reinhard global operator\n');

# specify resulting brightness of the tonampped image. See reinhardGlobal.py
# for details
a = 0.72;

# specify saturation of the resulting tonemapped image. See reinhardGlobal.py
# for details
saturation = 0.6;
[showGlobal, writeGlobal] = reinhardGlobal(hdrMap, a, saturation);

# create a window for display.
cv2.imshow("reinhardGlobal", showGlobal);
cv2.imwrite("reinhardGlobal.jpg", writeGlobal);
cv2.imshow("reinhardLocal", showLocal)
cv2.imwrite("reinhardLocal.jpg", writeLocal);
print('Finished! Enjoy...\n');
cv2.waitKey(0)
