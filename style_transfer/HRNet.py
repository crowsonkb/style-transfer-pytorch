import torch
import torch.nn as nn
import torchvision
import numpy as np

IN_MOMENTUM = 0.1


class ReflectionConv(nn.Module):
    '''
        Reflection padding convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ReflectionConv, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        return out


class ConvLayer(nn.Module):
    '''
        zero-padding convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        conv_padding = int(np.floor(kernel_size / 2))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding)
    def forward(self, x):
        return self.conv(x)
  

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True) # 1

        self.identity_block = nn.Sequential(
            ConvLayer(in_channels, out_channels // 4, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_channels // 4, momentum=IN_MOMENTUM),
            nn.ReLU(),
            ConvLayer(out_channels // 4, out_channels // 4, kernel_size, stride=stride),
            nn.InstanceNorm2d(out_channels // 4, momentum=IN_MOMENTUM),
            nn.ReLU(),
            ConvLayer(out_channels // 4, out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_channels, momentum=IN_MOMENTUM),
            nn.ReLU(),
        )
        self.shortcut = nn.Sequential(
            ConvLayer(in_channels, out_channels, 1, stride),
            nn.InstanceNorm2d(out_channels)
        )
    
    def forward(self, x):
        out = self.identity_block(x)
        if self.in_channels == self.out_channels:
            residual = x
        else: 
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class Upsample(nn.Module):
    '''
        Since the number of channels of the feature map changes after upsampling in HRNet.
        we have to write a new Upsample class.
    '''
    def __init__(self, in_channels, out_channels, scale_factor, mode):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.instance = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.upsample(out)
        out = self.instance(out)
        out = self.relu(out)
        
        return out


class HRNet(nn.Module):
    def __init__(self):
        super(HRNet, self).__init__()

        self.pass1_1 = BasicBlock(3, 16, kernel_size=3, stride=1)
        self.pass1_2 = BasicBlock(16, 32, kernel_size=3, stride=1)
        self.pass1_3 = BasicBlock(32, 32, kernel_size=3, stride=1)
        self.pass1_4 = BasicBlock(64, 64, kernel_size=3, stride=1)
        self.pass1_5 = BasicBlock(192, 64, kernel_size=3, stride=1)
        self.pass1_6 = BasicBlock(64, 32, kernel_size=3, stride=1)
        self.pass1_7 = BasicBlock(32, 16, kernel_size=3, stride=1)
        self.pass1_8 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.pass2_1 = BasicBlock(32, 32, kernel_size=3, stride=1)
        self.pass2_2 = BasicBlock(64, 64, kernel_size=3, stride=1)
        
        self.downsample1_1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.downsample1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.downsample1_3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.downsample1_4 = nn.Conv2d(32, 32, kernel_size=3, stride=4, padding=1)
        self.downsample1_5 = nn.Conv2d(64, 64, kernel_size=3, stride=4, padding=1)
        self.downsample2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.downsample2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.upsample1_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample1_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2_1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample2_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        map1 = self.pass1_1(x)
        map2 = self.pass1_2(map1)
        map3 = self.downsample1_1(map1)
        map4 = torch.cat((self.pass1_3(map2), self.upsample1_1(map3)), 1)
        map5 = torch.cat((self.downsample1_2(map2), self.pass2_1(map3)), 1)
        map6 = torch.cat((self.downsample1_4(map2), self.downsample2_1(map3)), 1)
        map7 = torch.cat((self.pass1_4(map4), self.upsample1_2(map5), self.upsample2_1(map6)), 1)
        out = self.pass1_5(map7)
        out = self.pass1_6(out)
        out = self.pass1_7(out)
        out = self.pass1_8(out)

        return out