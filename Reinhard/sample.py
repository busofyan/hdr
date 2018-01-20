import numpy as N
N.set_printoptions(threshold=N.nan)

def sample(image, sample_indices):

    # Takes the relevant samples of the input image I am not sure.... (matmul??)

    red_channel = image[:, :, 0]

    #green_channel = image[:, :, 2]
    #green = N.matmul(green_channel, sample_indices)

    #blue_channel = image[:, :, 3]
    #blue = N.matmul(blue_channel, sample_indices)

    red = 1
    green = 2
    blue = 3
    return [red, green, blue]
