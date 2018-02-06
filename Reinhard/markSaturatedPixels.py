import numpy as N
import cv2

def markSaturatedPixels(saturatedPixels, image):
    image = N.where(image == 255, 0, 1)
    saturatedPixels = N.where((image+saturatedPixels < 2) , 0, 1)

    return [saturatedPixels];