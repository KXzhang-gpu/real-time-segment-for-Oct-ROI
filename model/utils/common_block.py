# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_bn_relu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 has_bn=True,
                 has_relu=True,
                 bias=True,
                 groups=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias,
                              groups=groups)
        if has_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.bn = None

        if has_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def depthwise_xcorr(x, z):
    r"""
    Cross-correlation for Siamese Track

    Parameters
    ----------
    x : torch.Tensor
        feature_x from search image
    z : torch.Tensor
        feature_z from template image

    Returns
    -------
    torch.Tensor
        depth-wise cross correlation results
    """
    batch = z.shape[0]
    channel = z.shape[1]
    x = x.reshape(1, batch * channel, x.shape[2], x.shape[3])
    z = z.reshape(batch * channel, 1, z.shape[2], z.shape[3])
    out = F.conv2d(x, z, groups=batch * channel)
    out = out.reshape(batch, channel, out.shape[2], out.shape[3])
    return out
