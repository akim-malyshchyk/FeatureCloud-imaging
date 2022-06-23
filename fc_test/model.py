import torch.nn as nn
import torch.nn.functional as F


class ShortConvBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ShortConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.bn1(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BigConvBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BigConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.bn1(self.conv4(out))
        out = F.relu(out)
        return out


class VGG16(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(ShortConvBlock(in_channels, 64), ShortConvBlock(64, 64))
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer2 = nn.Sequential(BigConvBlock(64, 128), BigConvBlock(128, 128))
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer3 = nn.Sequential(BigConvBlock(128, 256))

        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.mp1(x)
        x = self.layer2(x)
        x = self.mp2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
