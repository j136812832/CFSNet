from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .block import *

####################
# Useful blocks
####################
class ResBlockDown(nn.Module):
    def __init__(
            self, conv, n_feats1, n_feats2,  n_feats3, n_feats4, kernel_size, padding=1,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlockDown, self).__init__()
        m = []
        m.append(nn.MaxPool2d(2))
        m.append(conv(n_feats1, n_feats2, kernel_size, padding, bias))
        m.append(nn.BatchNorm2d(n_feats2))
        m.append(act)
        m.append(conv(n_feats3, n_feats4, kernel_size, padding, bias))
        m.append(nn.BatchNorm2d(n_feats4))
        m.append(act)

        # for i in range(2):
        #     m.append(conv(n_feats, n_feats, kernel_size, padding=padding, bias=bias))
        #     if bn: m.append(nn.BatchNorm2d(n_feats))
        #     if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):

        res = self.body(x).mul(self.res_scale)

        # res = res + x

        return res

class ResBlockUp(nn.Module):
    def __init__(
            self, conv, n_feats1, n_feats2,  n_feats3, n_feats4, kernel_size, padding=1,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlockUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        m = []
        m.append(conv(n_feats1, n_feats2, kernel_size, padding, bias))
        m.append(nn.BatchNorm2d(n_feats2))
        m.append(act)
        m.append(conv(n_feats3, n_feats4, kernel_size, padding, bias))
        m.append(nn.BatchNorm2d(n_feats4))
        m.append(act)

        # for i in range(2):
        #     m.append(conv(n_feats, n_feats, kernel_size, padding=padding, bias=bias))
        #     if bn: m.append(nn.BatchNorm2d(n_feats))
        #     if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x1, x2):

        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        res = self.body(x).mul(self.res_scale)

        # res = res + x

        return res
# (1, 3, 256, 256) -> (1, 3, 256, 256)
class MainNet(nn.Module):
    def __init__(self, n_colors, out_nc, num_channels, num_blocks, task_type='sr', upscale=1):
        super(MainNet, self).__init__()

        self.task_type = task_type
        # define head
        self.head = default_conv(in_channels=n_colors, out_channels=num_channels,
                                 kernel_size=3, padding=1, bias=False, init_scale=0.1)

        # self.ResBlock1 = ResBlockDown(default_conv, 64, 128, 128, 128, kernel_size=3, act=nn.ReLU(True), res_scale=1)
        # self.ResBlock2 = ResBlockDown(default_conv, 128, 256, 256, 256, kernel_size=3, act=nn.ReLU(True), res_scale=1)
        # self.ResBlock3 = ResBlockDown(default_conv, 256, 512, 512, 512, kernel_size=3, act=nn.ReLU(True), res_scale=1)
        # self.ResBlock4 = ResBlockDown(default_conv, 512, 512, 512, 512, kernel_size=3, act=nn.ReLU(True), res_scale=1)
        # self.ResBlock5 = ResBlockUp(default_conv, 1024, 512, 512, 256, kernel_size=3, act=nn.ReLU(True), res_scale=1)
        # self.ResBlock6 = ResBlockUp(default_conv, 512, 256, 256, 128,  kernel_size=3, act=nn.ReLU(True), res_scale=1)
        # self.ResBlock7 = ResBlockUp(default_conv, 256, 128, 128, 64, kernel_size=3, act=nn.ReLU(True), res_scale=1)
        # self.ResBlock8 = ResBlockUp(default_conv, 128, 64, 64, 64, kernel_size=3, act=nn.ReLU(True), res_scale=1)
        # self.body = nn.ModuleList([self.ResBlock1, self.ResBlock2, self.ResBlock3, self.ResBlock4,
        #                            self.ResBlock5, self.ResBlock6, self.ResBlock7, self.ResBlock8])

        self.body = nn.ModuleList([ResBlockDown(default_conv, 64, 128, 128, 128, kernel_size=3, act=nn.ReLU(True), res_scale=1),
                                   ResBlockDown(default_conv, 128, 256, 256, 256, kernel_size=3, act=nn.ReLU(True), res_scale=1),
                                   ResBlockDown(default_conv, 256, 512, 512, 512, kernel_size=3, act=nn.ReLU(True), res_scale=1),
                                   ResBlockDown(default_conv, 512, 512, 512, 512, kernel_size=3, act=nn.ReLU(True), res_scale=1),
                                   ResBlockUp(default_conv, 1024, 512, 512, 256, kernel_size=3, act=nn.ReLU(True),
                                              res_scale=1),
                                   ResBlockUp(default_conv, 512, 256, 256, 128, kernel_size=3, act=nn.ReLU(True),
                                              res_scale=1),
                                   ResBlockUp(default_conv, 256, 128, 128, 64, kernel_size=3, act=nn.ReLU(True),
                                              res_scale=1),
                                   ResBlockUp(default_conv, 128, 64, 64, 64, kernel_size=3, act=nn.ReLU(True),
                                             res_scale=1)
                                   ])


        # self.body = nn.ModuleList(
        #     [ResBlock(default_conv,
        #               n_feats=num_channels, kernel_size=3, act=nn.ReLU(True), res_scale=1
        #               ) for _ in range(num_blocks)]
        # )
        self.end = default_conv(in_channels=num_channels, out_channels=out_nc,
                                kernel_size=3, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):
        output1 = self.head(x)
        output2 = torch.rand(1, 2, 3, 3)
        output3 = torch.rand(1, 2, 3, 3)
        output4 = torch.rand(1, 2, 3, 3)
        output5 = torch.rand(1, 2, 3, 3)


        head_f = output1
        output = output1


        count = 2

        for mbody in self.body:

            if(count == 6):
                output = mbody(output5, output4)
                count += 1
                continue
            if (count == 7):
                output = mbody(output, output3)
                count += 1
                continue
            if (count == 8):
                output = mbody(output, output2)
                count += 1
                continue
            if (count == 9):
                output = mbody(output, output1)
                count += 1
                continue

            output = mbody(output)
            if(count == 5):
                output5 = output
            if(count == 4):
                output4 = output
            if (count == 3):
                output3 = output
            if (count == 2):
                output2 = output

            count += 1


        output = self.end(output)
        return output

# tensor1 = torch.rand(1, 3, 256, 256)
# a = MainNet(3, 3, 64, 5)
# out = a(tensor1)
# print(out.shape)
# print(a)
# print(tensor1.shape[0])