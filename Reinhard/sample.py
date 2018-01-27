import numpy as N
N.set_printoptions(threshold=N.nan)

def sample(image, sample_indices):

    red_channel = image[:, :, 2];
    green_channel = image[:, :, 1];
    blue_channel = image[:, :, 0];

    red_channel = red_channel.flatten();
    blue_channel= blue_channel.flatten();
    green_channel =green_channel.flatten();
    red = red_channel[sample_indices ];
    blue =blue_channel[sample_indices];
    green =green_channel[sample_indices];

    return [red, green, blue]
