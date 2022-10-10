import torch
import torch.nn as nn
from .network import conv_layer


class FeatureExtractor2D(nn.Module):
    def __init__(self):
        super().__init__()
        base_channels = 8

        self.conv1 = nn.Sequential(
            conv_layer(3, base_channels * 2, 3, 1, True),
        )

        self.conv2 = nn.Sequential(
            conv_layer(3, base_channels * 2, 3, 2, True),
            conv_layer(base_channels * 2, base_channels * 2, 3, 1, True),
        )

        self.conv3 = nn.Sequential(
            conv_layer(3, base_channels * 2, 3, 2, True),
            conv_layer(base_channels * 2, base_channels * 2, 3, 1, True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 6, base_channels * 2, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, img_B3HW):
        feat1 = self.conv1(img_B3HW)
        feat2 = self.conv2(img_B3HW)
        feat3 = self.conv3(img_B3HW)
        feat4 = self.conv4(torch.cat([feat1, feat2, feat3], dim=1))
        return feat4