# https://github.com/milesial/Pytorch-UNet/tree/master/unet

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, pool_size=6):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_size),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, upsample_factor=6):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=True)
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


""" Full assembly of the parts to form the complete network """


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, scale_factor=6):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, pool_size=scale_factor)
        self.down2 = Down(128, 256, pool_size=scale_factor)
        self.down3 = Down(256, 512, pool_size=scale_factor)
        self.down4 = Down(512, 1024 // factor, pool_size=scale_factor)
        self.up1 = Up(1024, 512 // factor, bilinear, upsample_factor=scale_factor)
        self.up2 = Up(512, 256 // factor, bilinear, upsample_factor=scale_factor)
        self.up3 = Up(256, 128 // factor, bilinear, upsample_factor=scale_factor)
        self.up4 = Up(128, 64, bilinear, upsample_factor=scale_factor)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class SmallUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, scale_factor=6):
        super(SmallUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64, pool_size=scale_factor)
        self.down2 = Down(64, 128, pool_size=scale_factor)
        self.down3 = Down(128, 256, pool_size=scale_factor)
        self.down4 = Down(256, 512 // factor, pool_size=scale_factor)
        self.up1 = Up(512, 256 // factor, bilinear, upsample_factor=scale_factor)
        self.up2 = Up(256, 128 // factor, bilinear, upsample_factor=scale_factor)
        self.up3 = Up(128, 64 // factor, bilinear, upsample_factor=scale_factor)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        print(x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ExtraSmallUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, scale_factor=6):
        super(ExtraSmallUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64, pool_size=scale_factor)
        self.down2 = Down(64, 128, pool_size=scale_factor)
        self.down3 = Down(128, 256 // factor, pool_size=scale_factor)
        self.up1 = Up(256, 128 // factor, bilinear, upsample_factor=scale_factor)
        self.up2 = Up(128, 64 // factor, bilinear, upsample_factor=scale_factor)
        self.up3 = Up(64, 32, bilinear, upsample_factor=scale_factor)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #print(x4.shape)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class NanoUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, scale_factor=6):
        super(NanoUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32, pool_size=scale_factor)
        self.down2 = Down(32, 64, pool_size=scale_factor)
        self.down3 = Down(64, 128 // factor, pool_size=scale_factor)
        self.up1 = Up(128, 64 // factor, bilinear, upsample_factor=scale_factor)
        self.up2 = Up(64, 32 // factor, bilinear, upsample_factor=scale_factor)
        self.up3 = Up(32, 16, bilinear, upsample_factor=scale_factor)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


def small_segmentation_model_selector(model_name, n_classes, scale_factor):
    if "nano" in model_name:
        model = NanoUNet(n_channels=3, n_classes=n_classes, scale_factor=scale_factor)
    elif "extra_small" in model_name:
        model = ExtraSmallUNet(n_channels=3, n_classes=n_classes, scale_factor=scale_factor)
    elif "small" in model_name:
        model = SmallUNet(n_channels=3, n_classes=n_classes, scale_factor=scale_factor)
    elif "unet" in model_name:
        model = UNet(n_channels=3, n_classes=n_classes, scale_factor=scale_factor)
    else:
        assert False, f"Unknown model name: '{model_name}'"
    return model.cuda()
