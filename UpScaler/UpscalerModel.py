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


class ImageToImageDataset(Dataset):
  def __init__(self, input_dir: str, output_dir: str, train_input_transform, train_output_transform, train = True, test_transform = transforms.ToTensor()):
    self.input_paths = sorted(list(pathlib.Path(input_dir).glob("*.png")))
    self.output_paths = sorted(list(pathlib.Path(output_dir).glob("*.png")))
    self.transform_input = train_input_transform
    self.transform_output = train_output_transform
    self.transform_test = test_transform
    self.train = train

  def load_image(self, index: int) -> Tuple[Image.Image, Image.Image]:
    input_image_path = self.input_paths[index]
    output_image_path = self.output_paths[index]
    return Image.open(input_image_path), Image.open(output_image_path)

  def __len__(self) -> int:
    return len(self.input_paths)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    input_img, output_img = self.load_image(index)

    if self.train:
      return self.transform_input(input_img), self.transform_output(output_img)
    else:
      return self.transform_test(input_img), self.transform_test(output_img)


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

              # nn.UpsamplingNearest2d(scale_factor=2),
              # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
              # nn.LeakyReLU(0.2, inplace=True)
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


class Discriminator(nn.Module):
  def __init__(self, in_channels=3):
    super().__init__()

    def conv_block(input, output, stride):
      return nn.Sequential(
          nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1),
          nn.BatchNorm2d(output),
          nn.LeakyReLU(0.2, inplace=True)
      )

    self.model = nn.Sequential(
        conv_block(3, 64, 1),
        conv_block(64, 64, 2),
        conv_block(64, 128, 1),
        conv_block(128, 128, 2),
        conv_block(128, 256, 1),
        conv_block(256, 256, 2),
        conv_block(256, 512, 1),
        conv_block(512, 512, 2),
        nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
    )

  def forward(self, x):
    return self.model(x)


class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()
    # self.chosen_features = ['']
    self.model = models.vgg19(weights=True).features[:35]
    self.imagenet_norm = transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225])

  def forward(self, x):
    x = self.imagenet_norm(x)
    for layer_num, layer in enumerate(self.model):
      x = layer(x)
      if (layer_num == 34):
        return x