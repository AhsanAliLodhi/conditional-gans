import torch
import pdb
def add_one_hot_encoded_channel(input_image, no_of_channels, encoder_index,device ):
    '''
    :param input_image: input image
    :param no_of_channels: number of one hot encoded channels to be added to the input image
    :param encoder_index: index of channel to be set to 1s
    :return: concatenated input image with one hot encoding channels
    '''
    if no_of_channels == 0:
        return input_image
    else:
        if input_image.dim() == 3:
            extra_channels = torch.zeros((no_of_channels, input_image.shape[1], input_image.shape[2]))
            extra_channels[encoder_index, :, :] = extra_channels[encoder_index, :, :] + 1
            extra_channels = extra_channels.to(device)
            input_image = torch.cat((input_image, extra_channels), dim=0)

        if input_image.dim() == 4:
            extra_channels = torch.zeros((input_image.shape[0],no_of_channels, input_image.shape[2], input_image.shape[3]))
            extra_channels[0,encoder_index, :, :] += 1
            extra_channels = extra_channels.to(device)
            input_image = torch.cat((input_image, extra_channels), dim=1)

        return input_image