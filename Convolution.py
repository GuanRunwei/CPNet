import torch
import torchvision
import torch.nn as nn
import torch.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x




