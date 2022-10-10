import torch.nn as nn


def conv_layer(in_channels, out_channels, kernel_size, dilation, relu):
    stride = 1
    padding = (kernel_size - 1) // 2 * dilation
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
    return nn.Sequential(
        conv2d,
        nn.BatchNorm2d(out_channels),
        nn.ReLU() if relu else nn.Identity()
    )


def deconv_gn(in_channels, out_channels, kernel_size, stride, group_channels):
    padding = (kernel_size - stride) // 2
    group_num = max(1, out_channels // group_channels)
    
    # TODO: out_padding会不会导致特征图对不齐
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.GroupNorm(num_groups=group_num, num_channels=out_channels)
    )