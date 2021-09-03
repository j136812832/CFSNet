# -*- coding: UTF-8 -*-  
# Time : 2021/8/18--上午10:21
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .block_ed import *

####################
# Useful blocks
####################
# class ResBlock(nn.Module):
#     def __init__(
#             self, conv, n_feats, kernel_size, padding=1,
#             bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#
#         super(ResBlock, self).__init__()
#         m = []
#         for i in range(2):
#             m.append(conv(n_feats, n_feats, kernel_size, padding=padding, bias=bias))
#             if bn: m.append(nn.BatchNorm2d(n_feats))
#             if i == 0: m.append(act)
#
#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res = res + x
#
#         return res

class ResBlockDown(nn.Module):
    def __init__(
            self, conv, n_feats, o_feats, kernel_size, stride1=2, stride2=1, padding=1,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlockDown, self).__init__()
        m = []
        m.append(conv(n_feats, o_feats, kernel_size, stride1, padding, bias))
        # m.append(nn.BatchNorm2d(o_feats))
        m.append(act)
        m.append(conv(o_feats, o_feats, kernel_size, stride2, padding, bias))
        # m.append(nn.BatchNorm2d(o_feats))
        m.append(act)

        # for i in range(2):
        #     m.append(conv(n_feats, n_feats, kernel_size, padding=padding, bias=bias))
        #     if bn: m.append(nn.BatchNorm2d(n_feats))
        #     if i == 0: m.append(act)
        self.convZ = nn.Conv2d(n_feats, o_feats, 1, 2, bias=True)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        x = self.convZ(x)
        res = res + x

        return res
class ResBlockUp(nn.Module):
    def __init__(
            self, conv, n_feats, o_feats, kernel_size, stride=1, padding=1,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlockUp, self).__init__()

        m = []
        m.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        m.append(conv(n_feats, o_feats, kernel_size, stride, padding, True))
        # m.append(nn.BatchNorm2d(o_feats))
        m.append(act)
        m.append(conv(o_feats, o_feats, kernel_size, stride, padding, bias))
        # m.append(nn.BatchNorm2d(o_feats))
        m.append(act)

        # for i in range(2):
        #     m.append(conv(n_feats, n_feats, kernel_size, padding=padding, bias=bias))
        #     if bn: m.append(nn.BatchNorm2d(n_feats))
        #     if i == 0: m.append(act)
        # self.convZ = nn.Conv2d(n_feats, o_feats, 1, 2, bias=True)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        # x = self.convZ(x)
        res = res

        return res
class MainNet(nn.Module):
    def __init__(self, n_colors, out_nc, num_channels, num_blocks, task_type='sr', upscale=4):
        super(MainNet, self).__init__()

        # self.conv1_1 = default_conv(64, 128, 3, 2, 1, True)
        # self.bn1_1 = nn.BatchNorm2d(128)
        # self.conv1_2 = default_conv(128, 128, 3, 1, 1, True)
        # self.bn1_2 = nn.BatchNorm2d(128)
        #
        # self.conv2_1 = default_conv(128, 256, 3, 2, 1, True)
        # self.bn2_1 = nn.BatchNorm2d(256)
        # self.conv2_2 = default_conv(256, 256, 3, 1, 1, True)
        # self.bn2_2 = nn.BatchNorm2d(256)
        #
        # self.conv3_1 = default_conv(256, 512, 3, 2, 1, True)
        # self.bn3_1 = nn.BatchNorm2d(512)
        # self.conv3_2 = default_conv(512, 512, 3, 1, 1, True)
        # self.bn3_2 = nn.BatchNorm2d(512)
        #
        # self.conv4_1 = default_conv(512, 512, 3, 2, 1, True)
        # self.bn4_1 = nn.BatchNorm2d(512)
        # self.conv4_2 = default_conv(512, 512, 3, 1, 1, True)
        # self.bn4_2 = nn.BatchNorm2d(512)
        #
        # self.conv5_1 = default_conv(512, 256, 3, 1, 1, True)
        # self.bn5_1 = nn.BatchNorm2d(256)
        # self.conv5_2 = default_conv(256, 256, 3, 1, 1, True)
        # self.bn5_2 = nn.BatchNorm2d(256)
        #
        # self.conv6_1 = default_conv(256, 128, 3, 1, 1, True)
        # self.bn6_1 = nn.BatchNorm2d(128)
        # self.conv6_2 = default_conv(128, 128, 3, 1, 1, True)
        # self.bn6_2 = nn.BatchNorm2d(128)
        #
        # self.conv7_1 = default_conv(128, 64, 3, 1, 1, True)
        # self.bn7_1 = nn.BatchNorm2d(64)
        # self.conv7_2 = default_conv(64, 64, 3, 1, 1, True)
        # self.bn7_2 = nn.BatchNorm2d(64)
        #
        # self.conv8_1 = default_conv(64, 64, 3, 1, 1, True)
        # self.bn1_1 = nn.BatchNorm2d(64)
        # self.conv8_2 = default_conv(64, 64, 3, 1, 1, True)
        # self.bn1_1 = nn.BatchNorm2d(64)




        self.task_type = task_type
        # define head
        self.head = default_conv(in_channels=n_colors, out_channels=num_channels,
                                 kernel_size=3, stride=1, padding=1, bias=False, init_scale=0.1)
        # self.body = nn.ModuleList(
        #     [ResBlock(default_conv,
        #               n_feats=num_channels, kernel_size=3, act=nn.ReLU(True), res_scale=1
        #               ) for _ in range(num_blocks)]
        # )

        self.middle = default_conv(in_channels=512, out_channels=64,
                                 kernel_size=3, stride=1, padding=1, bias=False, init_scale=0.1)


        self.body = nn.ModuleList(
            [ResBlockDown(default_conv, n_feats=64, o_feats=128, kernel_size=3, stride1=2, stride2=1, act=nn.ReLU(True), res_scale=1),
             ResBlockDown(default_conv, n_feats=128, o_feats=256, kernel_size=3, stride1=2, stride2=1, act=nn.ReLU(True), res_scale=1),
             ResBlockDown(default_conv, n_feats=256, o_feats=512, kernel_size=3, stride1=2, stride2=1, act=nn.ReLU(True), res_scale=1),
             ResBlockDown(default_conv, n_feats=512, o_feats=512, kernel_size=3, stride1=2, stride2=1, act=nn.ReLU(True), res_scale=1),

             ResBlockDown(default_conv, n_feats=512, o_feats=64, kernel_size=3, stride1=1, stride2=1,  act=nn.ReLU(True), res_scale=1),
             ResBlockDown(default_conv, n_feats=64, o_feats=64, kernel_size=3, stride1=1, stride2=1,  act=nn.ReLU(True), res_scale=1),
             ResBlockDown(default_conv, n_feats=64, o_feats=64, kernel_size=3, stride1=1, stride2=1,  act=nn.ReLU(True), res_scale=1),
             ResBlockDown(default_conv, n_feats=64, o_feats=64, kernel_size=3, stride1=1, stride2=1,  act=nn.ReLU(True), res_scale=1),
             ResBlockDown(default_conv, n_feats=64, o_feats=64, kernel_size=3, stride1=1, stride2=1,  act=nn.ReLU(True), res_scale=1),
             ResBlockDown(default_conv, n_feats=64, o_feats=64, kernel_size=3, stride1=1, stride2=1,  act=nn.ReLU(True), res_scale=1),

             ResBlockUp(default_conv, n_feats=64, o_feats=64, kernel_size=3, act=nn.ReLU(True), res_scale=1),
             ResBlockUp(default_conv, n_feats=64, o_feats=64, kernel_size=3, act=nn.ReLU(True), res_scale=1),
             ResBlockUp(default_conv, n_feats=64, o_feats=64, kernel_size=3, act=nn.ReLU(True), res_scale=1),
             ResBlockUp(default_conv, n_feats=64, o_feats=64, kernel_size=3, act=nn.ReLU(True), res_scale=1), ]
        )

        if self.task_type == 'sr':
            self.tail = nn.Sequential(
                default_conv(in_channels=num_channels, out_channels=num_channels,
                             kernel_size=3, stride=1, padding=1, bias=False, init_scale=0.1),
                Upsampler(default_conv, upscale, num_channels, act=False),
            )

        self.end = default_conv(in_channels=num_channels, out_channels=out_nc,
                                kernel_size=3, stride=1, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):
        output = self.head(x)
        head_f = output
        for mbody in self.body:
            output = mbody(output)
        if self.task_type == 'sr':
            output = self.tail(output)
        output = self.end(output + head_f)
        return output


# tensor1 = torch.rand(1, 3, 256, 256)
# a = MainNet(3, 3, 64, 5)
# print(a)
# out = a(tensor1)
# print(out.shape)

# control_vector = torch.ones(2, 512) * 1
# print(control_vector.shape)
# print(type(control_vector))