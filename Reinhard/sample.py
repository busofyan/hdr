import numpy as N
N.set_printoptions(threshold=N.nan)

def sample(image, sample_indices):

    red_channel = image[:, :, 2];

    print('Picture Begin');
    red_channel = red_channel.flatten();
    print('sample_indices: ', sample_indices);
    red = red_channel[sample_indices - 1];
    print('red: ', red);
    print('Picture End');

    # print('Picture Begin');
    # print('redChannel: ', red_channel);
    # print('red: ', red);
    # print('Picture End');

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
