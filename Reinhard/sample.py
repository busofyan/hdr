import numpy as N

def sample(image, sample_indices, num_pixels):
    sample_indices = sample_indices.reshape((128));
    # print(sample_indices);

    red_channel = image[:, :, 2];
    green_channel = image[:, :, 1];
    blue_channel = image[:, :, 0];

    reddish = [];
    greenish = [];
    blueish = [];

    for i in range(0, image.shape[1]):
        for j in range(0, image.shape[0]):
            reddish.append(red_channel[j, i]);
            greenish.append(green_channel[j, i]);
            blueish.append(blue_channel[j, i]);

    red_channel = N.reshape(reddish, (num_pixels, 1));
    green_channel = N.reshape(greenish, (num_pixels, 1));
    blue_channel = N.reshape(blueish, (num_pixels, 1));

    red = red_channel[(sample_indices, 1 - 1)];
    green = green_channel[(sample_indices, 1 - 1)];
    blue = blue_channel[(sample_indices, 1 - 1)];


    #red_channel = image[:, :, 2];
    #red = red_channel.flatten();
    #red = red[sample_indices]

    #green_channel = image[:, :, 1];
    #green = green_channel.flatten();
    #green = green[sample_indices]

    #blue_channel = image[:, :, 0];
    #blue = blue_channel.flatten();
    #blue = blue[sample_indices]

    return [red, green, blue]
