
def sample(image, sample_indices, num_pixels):

    red_channel = image[:, :, 2];
    red = red_channel.flatten();
    red = red[sample_indices]

    green_channel = image[:, :, 1];
    green = green_channel.flatten();
    green = green[sample_indices]

    blue_channel = image[:, :, 0];
    blue = blue_channel.flatten();
    blue = blue[sample_indices]

    return [red, green, blue]
