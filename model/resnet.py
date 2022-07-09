import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels *
                               self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):

        identity = x.clone()

        x = self.conv1(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x = x + identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=num_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_0 = self.make_layer(ResBlock, layer_list[0], plane=64)
        self.layer_1 = self.make_layer(ResBlock, layer_list[1], plane=128)
        self.layer_2 = self.make_layer(ResBlock, layer_list[2], plane=256)
        self.layer_3 = self.make_layer(ResBlock, layer_list[3], plane=512)

        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = self.avg_pool2d(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return F.log_softmax(x, dim=1)

    def make_layer(self, ResBlock, block, plane, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != plane * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, plane * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(plane * ResBlock.expansion)

            )

        layers.append(ResBlock(self.in_channels, plane, i_downsample=ii_downsample, stride=stride))

        self.in_channels = plane * ResBlock.expansion

        for i in range(block - 1):
            layers.append(ResBlock(self.in_channels, plane))

        return nn.Sequential(*layers)


class ResNet50(ResNet):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__(BottleNeck, [3, 4, 6, 3], num_classes, 1)


model = ResNet50()
