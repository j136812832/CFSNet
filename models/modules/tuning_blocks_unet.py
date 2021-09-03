from torch import nn
import torch
import math
from .block import *

# class TuningBlock(nn.Module):
#     def __init__(self, input_size):
#         super(TuningBlock, self).__init__()
#         self.conv0 = default_conv(in_channels=input_size, out_channels=input_size,
#                                   kernel_size=3, padding=1, bias=False, init_scale=0.1)
#         self.relu0 = nn.ReLU()
#         self.conv1 = default_conv(in_channels=input_size, out_channels=input_size,
#                                   kernel_size=3, padding=1, bias=False, init_scale=0.1)
#
#     def forward(self, x):
#         out = self.conv0(x)
#         out = self.relu0(out)
#         out = self.conv1(out)
#         return out

class TuningBlockUnetDown(nn.Module):
    def __init__(self, input_size, output_size):
        super(TuningBlockUnetDown, self).__init__()
        self.maxpooling = nn.MaxPool2d(2)
        self.conv0 = default_conv(in_channels=input_size, out_channels=output_size,
                                  kernel_size=3, padding=1, bias=False, init_scale=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = default_conv(in_channels=output_size, out_channels=output_size,
                                  kernel_size=3, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):
        out = self.maxpooling(x)
        out = self.conv0(out)
        out = self.relu0(out)
        out = self.conv1(out)
        return out
class TuningBlockUnetUp(nn.Module):
    def __init__(self, input_size, output_size):
        super(TuningBlockUnetUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0 = default_conv(in_channels=input_size, out_channels=output_size,
                                  kernel_size=3, padding=1, bias=False, init_scale=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = default_conv(in_channels=output_size, out_channels=output_size,
                                  kernel_size=3, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):
        out = self.up(x)
        out = self.conv0(out)
        out = self.relu0(out)
        out = self.conv1(out)
        return out

class TuningBlockModule(nn.Module):
    def __init__(self, channels=64, num_blocks=5, task_type='sr', upscale=1):
        super(TuningBlockModule, self).__init__()
        self.num_channels = channels
        self.task_type = task_type
        # define control variable
        self.control_alpha = nn.Sequential(
            default_Linear(512, 256, bias=False),
            default_Linear(256, 128, bias=False),
            default_Linear(128, channels, bias=False)
        )
        # self.adaptive_alpha = nn.ModuleList(
        #     [nn.Sequential(
        #         default_Linear(channels, channels, bias=False),
        #         default_Linear(channels, channels, bias=False)
        #     ) for _ in range(num_blocks)]
        # )
        self.adaptive_alpha_unet = nn.ModuleList(
            [   nn.Sequential(
                    default_Linear(64, 128, bias=False),
                    default_Linear(128, 128, bias=False)),
                nn.Sequential(
                    default_Linear(64, 128, bias=False),
                    default_Linear(128, 256, bias=False)),
                nn.Sequential(
                    default_Linear(64, 256, bias=False),
                    default_Linear(256, 512, bias=False)),
                nn.Sequential(
                    default_Linear(64, 256, bias=False),
                    default_Linear(256, 512, bias=False)),
                nn.Sequential(
                    default_Linear(64, 128, bias=False),
                    default_Linear(128, 256, bias=False)),
                nn.Sequential(
                    default_Linear(64, 128, bias=False),
                    default_Linear(128, 128, bias=False)),
                nn.Sequential(
                    default_Linear(64, 64, bias=False),
                    default_Linear(64, 64, bias=False)),
                nn.Sequential(
                    default_Linear(64, 64, bias=False),
                    default_Linear(64, 64, bias=False))
            ]
        )
        # self.tuning_blocks = nn.ModuleList(
        #     [TuningBlock(channels) for _ in range(num_blocks)]
        # )

        self.tuning_blocks_unet = nn.ModuleList(
            [TuningBlockUnetDown(64, 128),
             TuningBlockUnetDown(128, 256),
             TuningBlockUnetDown(256, 512),
             TuningBlockUnetDown(512, 512),
             TuningBlockUnetUp(512, 256),
             TuningBlockUnetUp(256, 128),
             TuningBlockUnetUp(128, 64),
             TuningBlockUnetUp(64, 64),
             ]
        )
        # if self.task_type == 'sr':
        #     self.tuning_blocks_unet.append(nn.Sequential(
        #         default_conv(in_channels=channels, out_channels=channels,
        #                      kernel_size=3, padding=1, bias=False, init_scale=0.1),
        #         Upsampler(default_conv, upscale, channels, bias=False, act=False),
        #     ))
        #     self.adaptive_alpha_unet.append(nn.Sequential(
        #         default_Linear(channels, channels, bias=False),
        #         default_Linear(channels, channels, bias=False)
        #     ))

    def forward(self, x, alpha, number=0):
        input_alpha = self.control_alpha(alpha)
        # print("input_alpha:{}".format(input_alpha.shape))
        tuning_f = self.tuning_blocks_unet[number](x)
        # print("tuning_f:{}".format(tuning_f.shape))
        ad_alpha = self.adaptive_alpha_unet[number](input_alpha)
        # print("ad_alpha:{}".format(ad_alpha.shape))
        ad_alpha = ad_alpha.view(-1, tuning_f.shape[1], 1, 1)
        # print("ad_alpha:{}".format(ad_alpha.shape))
        return tuning_f * ad_alpha, torch.ones_like(ad_alpha).cuda()-ad_alpha

# tensor1 = torch.rand(1, 3, 256, 256)
# a = TuningBlockModule()
# print(a)
