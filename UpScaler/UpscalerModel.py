import os
import pathlib
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class DenseBlock(nn.Module):
  def __init__(self, in_channels, out_channels = 32):
    super().__init__()
    self.layers = nn.ModuleList()
    self.in_channels = in_channels
    self.out_channels = out_channels

    for i in range(5):
      self.layers.append(
          nn.Sequential(
              nn.Conv2d(self.in_channels + i * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
              nn.LeakyReLU(0.2, inplace=True)
          )
      )

    self.conv1 = nn.Conv2d(self.in_channels + 5 * self.out_channels, self.in_channels, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    features = [x]
    for layer in self.layers:
      out = layer(torch.cat(features, dim=1))
      features.append(out)
    out = self.conv1(torch.cat(features, dim=1))
    return out


class RRDB(nn.Module):  #Residual-in-Residual Dense Block
  def __init__(self, in_channels, out_channels = 32, beta=0.2):
    super().__init__()
    self.beta = beta
    self.block1 = DenseBlock(64, 32)
    self.block2 = DenseBlock(64, 32)
    self.block3 = DenseBlock(64, 32)

  def forward(self, x):
    x = self.block1(x) * self.beta + x
    x = self.block2(x) * self.beta + x
    x = self.block3(x) * self.beta + x
    return x


class Generator(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv_initial = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
    self.RRDB_sequence = nn.ModuleList()

    for i in range(8):
      self.RRDB_sequence.append(
          RRDB(64, 32, 0.2)
      )

    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    self.Upsampling = nn.Sequential(
              nn.UpsamplingNearest2d(scale_factor=2),
              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
              nn.LeakyReLU(0.2, inplace=True),
          )

    self.image_reconstruction = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
    )

  def forward(self, x):
    x = self.conv_initial(x)
    out = x
    for layer in self.RRDB_sequence:
      out = layer(out)
    out = self.conv2(out)
    x = x + out
    x = self.Upsampling(x)
    x = self.image_reconstruction(x)
    return x
