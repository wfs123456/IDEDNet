# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 10:46
# @Author  : Fusen Wang
# @Email   : 201924131014@cqu.edu.cn
# @File    : IDEDNet.py
# @Software: PyCharm
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import torch
import os

def conv(in_channel, out_channel, kernel_size, dilation=1, bn=True):
    padding = dilation # maintain the previous size
    if bn:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, dilation=dilation,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True)
        )

def make_layers(cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

class DownModule(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DownModule, self).__init__()

        self.Dconv = nn.Sequential(conv(in_channel, out_channel, 3),
                                   nn.AvgPool2d(2, stride=2),
                                   conv(out_channel, out_channel, 3)
                                   )
        self.init_param()
    def forward(self, x):
        x = self.Dconv(x)
        return x

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# class Interpolate(nn.Module):
#     def __init__(self,scale_factor,mode):
#         super(Interpolate,self).__init__()
#         self.interpolate = F.interpolate
#         self.scale_factor = scale_factor
#         self.mode = mode
#
#     def forward(self, x):
#         x = self.interpolate(x,scale_factor=self.scale_factor,mode=self.mode)
#         return x

class UpModule(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(UpModule, self).__init__()

        self.Tconv = nn.ConvTranspose2d(in_channel, in_channel, 2, stride=2)
        # self.upsample = Interpolate(2, 'nearest')
        self.Uconv = nn.Sequential(conv(in_channel, out_channel, 3),
                                   conv(out_channel, out_channel, 3))

        self.init_param()
    def forward(self, x):
        x = self.Tconv(x)
        #print("11", x.size())
        x = self.Uconv(x)
        return x

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.down1 = DownModule(1, 32)
        self.down2 = DownModule(32, 64)
        self.down3 = DownModule(64, 128)
        self.drop3 = nn.Dropout2d(0.2)
        self.down4 = DownModule(128, 256)
        self.drop4 = nn.Dropout2d(0.2)
        self.down5 = DownModule(256, 512)

        self.FC = nn.Sequential(nn.Conv2d(512, 512, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, 1))

        self.up1 = UpModule(512, 256)
        self.up2 = UpModule(256, 128)
        self.up3 = UpModule(128, 64)
        self.up4 = UpModule(64, 32)
        self.up5 = UpModule(32, 1)

        self.init_param()

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # print("loading pretrained vgg16_bn!")
        # if os.path.exists("weights/vgg16_bn.pth"):
        #     print("find pretrained weights!")
        #     vgg16_bn = models.vgg16_bn(pretrained=False)
        #     vgg16_weights = torch.load("weights/vgg16_bn.pth")
        #     vgg16_bn.load_state_dict(vgg16_weights)
        # else:
        #     vgg16_bn = models.vgg16_bn(pretrained=True)
        #
        # # load vgg16_bn weights
        # self.FEC.load_state_dict(vgg16_bn.features[:33].state_dict())

    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3_1 = self.drop3(x3)
        x4 = self.down4(x3_1)
        # print("x4", x4.size())
        x4_1 = self.drop4(x4)
        x5 = self.down5(x4_1)
        # print('x5',x5.size())

        x6 = self.FC(x5)
        #print('x6',x6.size())

        x = self.up1(x6+x5)
        x = self.up2(x+x4)
        x = self.up3(x+x3)
        x = self.up4(x+x2)
        x = self.up5(x+x1)
        return x

if __name__ == "__main__":
    from thop import profile
    from thop import clever_format
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    net = Net()
    x = torch.ones((16, 1, 32, 32))
    print(x.size())
    y = net(x)
    print(y.size())
    # flops, params = profile(net, inputs=(x, ))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)