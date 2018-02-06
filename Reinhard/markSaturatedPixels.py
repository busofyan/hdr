import numpy as N
import cv2

def markSaturatedPixels(saturatedPixels, saturatedPixelsRed):
    tempValues = [];

    # copy matrix axis values to array
    for i in range(0, saturatedPixels.shape[2]):
        for j in range(0, saturatedPixels.shape[1]):
            tempValues.append(saturatedPixels[2, :, :][j, i]);

    for i in range(0, tempValues.__len__()):
        for j in range(0, saturatedPixelsRed.__len__()):
            if i == saturatedPixelsRed[j]:
                tempValues[i] = 0;

    k = 0;
    for i in range(0, saturatedPixels.shape[2]):
        for j in range(0, saturatedPixels.shape[1]):
            saturatedPixels[2, :, :][j, i] = tempValues[k];
            saturatedPixels[1, :, :][j, i] = tempValues[k];
            saturatedPixels[0, :, :][j, i] = tempValues[k];
            k = k + 1;

    return [saturatedPixels];