import torch
from torch import nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ConvBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv = conv3x3(inplanes, planes, stride)
        self.bn = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out
    

class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)

        return out


class ResModule(nn.Module):

    def __init__(self, inplanes, planes):
        super().__init__()
        self.res1 = ResBlock(inplanes, planes)
        self.res2 = ResBlock(planes, planes)
        self.res3 = ResBlock(planes, planes)
        self.res4 = ResBlock(planes, planes)

    def forward(self, x):
        residual = x

        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        out = out + residual

        return out


class Down(nn.Module):

    def __init__(self, inplanes, planes, downscale_factor):
        super().__init__()
        self.pool = nn.MaxPool2d(downscale_factor)
        self.conv = ConvBlock(inplanes, planes)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out)

        return out


class Up(nn.Module):

    def __init__(self, inplanes, stride=1):
        super().__init__()
        self.conv = ConvBlock(inplanes, inplanes * 4)
        self.up = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.up(out)

        return out


class TPUNet(nn.Module):

    def __init__(self, mid_ch=50):
        super().__init__()
        self.conv0 = ConvBlock(inplanes=2, planes=mid_ch)
        
        self.down1 = Down(inplanes=mid_ch, planes=mid_ch, downscale_factor=2)
        self.down2 = Down(inplanes=mid_ch, planes=mid_ch, downscale_factor=4)
        self.down3 = Down(inplanes=mid_ch, planes=mid_ch, downscale_factor=8)

        self.res = ResModule(inplanes=mid_ch, planes=mid_ch)

        self.up = Up(inplanes=mid_ch)

        self.conv1 = ConvBlock(inplanes=mid_ch, planes=mid_ch)
        self.conv2 = ConvBlock(inplanes=mid_ch * 4, planes=1)
        
    def forward(self, x):
        x0 = self.conv0(x)

        x1 = self.down1(x0)
        x1 = self.res(x1)

        x2 = self.down2(x0)
        x2 = self.res(x2)

        x3 = self.down3(x0)
        x3 = self.res(x3)

        x0 = self.res(x0)

        x3 = self.up(x3)
        x3 = self.up(x3)
        x3 = self.up(x3)
        x3 = self.conv1(x3)

        x2 = self.up(x2)
        x2 = self.up(x2)
        x2 = self.conv1(x2)

        x1 = self.up(x1)
        x1 = self.conv1(x1)

        x0 = self.conv1(x1)

        out = torch.cat([x0, x1, x2, x3], dim=1)
        out = self.conv2(out)
        
        return out
