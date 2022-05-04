import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
from Convolution import BasicConv
from MHSA import Attention
from MHSA import PatchEmbed
import torchsummary
from einops import rearrange, reduce, repeat
import math

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_heads=4,
                 dropout_ratio=0.05):
        super().__init__()
        self.conv1 = BasicConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, stride=1)
        self.mish1 = nn.Mish()
        self.conv2 = BasicConv(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, stride=1)
        self.mish2 = nn.Mish()
        self.mhsa = Attention(dim=out_channels, num_heads=num_heads)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)
        patch_embed = PatchEmbed(input_shape=x.shape[-2:], num_features=x.shape[1], patch_size=2,
                                 in_chans=x.shape[1]).to(device)
        x = patch_embed(x)

        x = self.mhsa(x)
        x = rearrange(x, 'b (w h) c -> b w h c', w=int(math.sqrt(x.shape[1])))
        x = x.permute(0, 3, 1, 2)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class CPNet(nn.Module):
    def __init__(self, in_channels=3, kernel_size=3, padding=1, num_heads=4,
                 dropout_ratio=0.01, num_classes=1000):
        super().__init__()
        self.basic_conv_1 = BasicConv(in_channels=in_channels, out_channels=64)
        self.basic_conv_2 = BasicConv(in_channels=64, out_channels=128)
        self.ra_cell_1 = ResidualBlock(in_channels=128, out_channels=256)
        self.ra_cell_2 = ResidualBlock(in_channels=256, out_channels=512)
        self.ra_cell_3 = ResidualBlock(in_channels=512, out_channels=1024)
        self.downsampling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.basic_conv_1(x)
        x = self.basic_conv_2(x)
        x = self.ra_cell_1(x)
        x = self.ra_cell_2(x)
        x = self.ra_cell_3(x)
        x = self.downsampling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

