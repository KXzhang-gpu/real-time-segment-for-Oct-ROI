# -*- coding: UTF-8 -*-
import torch
from torch import nn

from model.utils.common_block import conv_bn_relu


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = conv_bn_relu(1, 96, kernel_size=11, stride=2, padding=0)
        self.pool1 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv2 = conv_bn_relu(96, 256, 5, 1, 0)
        self.pool2 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv3 = conv_bn_relu(256, 384, 3, 1, 0)
        self.conv4 = conv_bn_relu(384, 384, 3, 1, 0)
        self.conv5 = conv_bn_relu(384, 256, 3, 1, 0, has_relu=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
