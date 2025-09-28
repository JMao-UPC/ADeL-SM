import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution Block: Conv2d -> ReLU -> Conv2d -> ReLU.
    Maintains spatial resolution (using padding=1) and increases channel depth.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # In-place activation to save memory
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Image Restoration.
    Encoder (downsampling) + Decoder (upsampling) with skip connections.

    Input: 1-channel grayscale image (H x W).
    Output: 1-channel restored image (same size as input).
    """

    def __init__(self):
        super().__init__()
        # Encoder (Downsampling)
        self.double_conv1 = DoubleConv(1, 32)  # Input: 1ch -> Output: 32ch
        self.max_pool1 = nn.MaxPool2d(2)  # Downsample by 2x (H/2 x W/2)

        self.double_conv2 = DoubleConv(32, 64)  # Input: 32ch -> Output: 64ch
        self.max_pool2 = nn.MaxPool2d(2)  # Downsample by 2x (H/4 x W/4)

        self.bottleneck = DoubleConv(64, 128)  # Bottleneck: 64ch -> 128ch (H/4 x W/4)

        # Decoder (Upsampling)
        self.up_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Upsample 2x (H/2 x W/2)
        self.double_conv3 = DoubleConv(128, 64)  # Input: 64ch (up) + 64ch (skip) = 128ch -> Output: 64ch

        self.up_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # Upsample 2x (H x W)
        self.double_conv4 = DoubleConv(64, 32)  # Input: 32ch (up) + 32ch (skip) = 64ch -> Output: 32ch

        # Output Layer (restore 1 channel)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.double_conv1(x)  # Skip connection 1: H x W x 32
        pool1 = self.max_pool1(conv1)  # H/2 x W/2 x 32

        conv2 = self.double_conv2(pool1)  # Skip connection 2: H/2 x W/2 x 64
        pool2 = self.max_pool2(conv2)  # H/4 x W/4 x 64

        # Bottleneck
        bottleneck = self.bottleneck(pool2)  # H/4 x W/4 x 128

        # Decoder
        up2 = self.up_conv2(bottleneck)  # H/2 x W/2 x 64
        # Concatenate with skip connection 2 (align spatial size)
        concat2 = torch.cat([up2, conv2], dim=1)  # H/2 x W/2 x 128
        conv3 = self.double_conv3(concat2)  # H/2 x W/2 x 64

        up1 = self.up_conv1(conv3)  # H x W x 32
        # Concatenate with skip connection 1
        concat1 = torch.cat([up1, conv1], dim=1)  # H x W x 64
        conv4 = self.double_conv4(concat1)  # H x W x 32

        # Output
        out = self.out_conv(conv4)  # H x W x 1
        return out