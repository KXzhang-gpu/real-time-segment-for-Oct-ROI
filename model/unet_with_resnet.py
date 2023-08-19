import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_block(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1, use_conv_1=False):
        super(Residual_block, self).__init__()
        self.use_conv_1 = use_conv_1
        if self.use_conv_1:
            self.conv3 = conv_bn_relu(input_channels,
                                      num_channels,
                                      stride=strides,
                                      kszie=1,
                                      pad=0,
                                      has_bn=True,
                                      has_relu=False,
                                      bias=False)
        else:
            self.conv3 = None
        self.conv1 = conv_bn_relu(input_channels,
                                  num_channels,
                                  stride=strides,
                                  kszie=3,
                                  pad=1,
                                  has_bn=True,
                                  has_relu=True,
                                  bias=False)
        self.conv2 = conv_bn_relu(num_channels,
                                  num_channels,
                                  stride=1,
                                  kszie=3,
                                  pad=1,
                                  has_bn=True,
                                  has_relu=False,
                                  bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        if self.use_conv_1:
            residual = self.conv3(residual)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.relu(x)
        return x


class conv_bn_relu(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 kszie=3,
                 pad=0,
                 has_bn=True,
                 has_relu=True,
                 bias=True,
                 groups=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kszie,
                              stride=stride,
                              padding=pad,
                              bias=bias,
                              groups=groups)

        if has_bn:
            self.bn = nn.BatchNorm2d(out_channel)
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


class UNet(nn.Module):
    def __init__(self, n_classes=1, block=Residual_block, bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        self.block = block
        self.down1 = nn.Sequential(nn.Sequential(
            conv_bn_relu(1,
                         32,
                         stride=2,
                         kszie=7,
                         pad=3,
                         has_bn=True,
                         has_relu=True,
                         bias=False),
            conv_bn_relu(32,
                         32,
                         stride=1,
                         kszie=3,
                         pad=1,
                         has_bn=True,
                         has_relu=True,
                         bias=False),
            conv_bn_relu(32,
                         32,
                         stride=1,
                         kszie=3,
                         pad=1,
                         has_bn=True,
                         has_relu=True,
                         bias=False)))
        self.maxpool = nn.MaxPool2d(3, 2, 1, ceil_mode=False)
        self.down2 = nn.Sequential(self._resnet_block(self.block, 32, 64, 2, 1))
        self.down3 = nn.Sequential(self._resnet_block(self.block, 64, 128, 2, 2))
        self.down4 = nn.Sequential(self._resnet_block(self.block, 128, 256, 2, 2))
        self.down5 = nn.Sequential(self._resnet_block(self.block, 256, 512, 2, 2))
        factor = 2 if bilinear else 1
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = OutConv(32, n_classes)

    def _resnet_block(self, block, input_channels, num_channels, num_residuals, strides):
        stage = []
        stage.append(block(input_channels, num_channels, strides, True))
        for i in range(1, num_residuals):
            stage.append(block(num_channels, num_channels, 1, False))
        return nn.Sequential(*stage)

    def forward(self, x):
        x1 = self.down1(x)
        x1_new = self.maxpool(x1)
        x2 = self.down2(x1_new)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    X = torch.rand(size=(1, 1, 512, 512))
    net = UNet(1)
    X_new = net(X)
    print(X_new.shape)
