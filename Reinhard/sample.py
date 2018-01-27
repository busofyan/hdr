import numpy as N
N.set_printoptions(threshold=N.nan)

def sample(image, sample_indices):

    print(sample_indices);
    red_channel = image[:, :, 2];
    print(red_channel);
    red = red_channel[[0, 1, 2], [0]];

    # print('Picture Begin');
    # print(red_channel);
    # print('Picture End');

    #green_channel = image[:, :, 2]
    #green = N.matmul(green_channel, sample_indices)

    #blue_channel = image[:, :, 3]
    #blue = N.matmul(blue_channel, sample_indices)

    red = 1
    green = 2
    blue = 3
    return [red, green, blue]
