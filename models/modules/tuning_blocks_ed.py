from torch import nn
import torch
import math
from .block_ed import *

class TuningBlockDown(nn.Module):
    def __init__(self, input_size, output_size, stride1, stride2):
        super(TuningBlockDown, self).__init__()
        self.conv0 = default_conv(in_channels=input_size, out_channels=output_size,
                                  kernel_size=3, stride=stride1, padding=1, bias=False, init_scale=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = default_conv(in_channels=output_size, out_channels=output_size,
                                  kernel_size=3, stride=stride2, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu0(out)
        out = self.conv1(out)
        return out

class TuningBlockUp(nn.Module):
    def __init__(self, input_size, output_size):
        super(TuningBlockUp, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0 = default_conv(in_channels=input_size, out_channels=output_size,
                                  kernel_size=3, stride=1, padding=1, bias=False, init_scale=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = default_conv(in_channels=output_size, out_channels=output_size,
                                  kernel_size=3, stride=1, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):

        x = self.up(x)
        out = self.conv0(x)
        out = self.relu0(out)
        out = self.conv1(out)
        return out

class TuningBlockModule(nn.Module):
    def __init__(self, channels=64, num_blocks=5, task_type='sr', upscale=4):
        super(TuningBlockModule, self).__init__()
        self.num_channels = channels
        self.task_type = task_type
        # define control variable
        self.control_alpha = nn.Sequential(
            default_Linear(512, 256, bias=False),
            default_Linear(256, 128, bias=False),
            default_Linear(128, channels, bias=False)
        )
        self.adaptive_alpha = nn.ModuleList(
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
                    default_Linear(64, 64, bias=False),
                    default_Linear(64, 64, bias=False)),
                nn.Sequential(
                    default_Linear(64, 64, bias=False),
                    default_Linear(64, 64, bias=False)),
                nn.Sequential(
                    default_Linear(64, 64, bias=False),
                    default_Linear(64, 64, bias=False)),
                nn.Sequential(
                    default_Linear(64, 64, bias=False),
                    default_Linear(64, 64, bias=False)),
                nn.Sequential(
                    default_Linear(64, 64, bias=False),
                    default_Linear(64, 64, bias=False)),
                nn.Sequential(
                    default_Linear(64, 64, bias=False),
                    default_Linear(64, 64, bias=False)),

                nn.Sequential(
                    default_Linear(64, 64, bias=False),
                    default_Linear(64, 64, bias=False)),
                nn.Sequential(
                    default_Linear(64, 64, bias=False),
                    default_Linear(64, 64, bias=False)),
                nn.Sequential(
                    default_Linear(64, 64, bias=False),
                    default_Linear(64, 64, bias=False)),
                nn.Sequential(
                    default_Linear(64, 64, bias=False),
                    default_Linear(64, 64, bias=False))
            ]
        )
        self.tuning_blocks = nn.ModuleList(
            [TuningBlockDown(64, 128, 2, 1),
             TuningBlockDown(128, 256, 2, 1),
             TuningBlockDown(256, 512, 2, 1),
             TuningBlockDown(512, 512, 2, 1),

             TuningBlockDown(512, 64, 1, 1),
             TuningBlockDown(64, 64, 1, 1),
             TuningBlockDown(64, 64, 1, 1),
             TuningBlockDown(64, 64, 1, 1),
             TuningBlockDown(64, 64, 1, 1),
             TuningBlockDown(64, 64, 1, 1),

             TuningBlockUp(64, 64),
             TuningBlockUp(64, 64),
             TuningBlockUp(64, 64),
             TuningBlockUp(64, 64)
             ]
        )
        if self.task_type == 'sr':
            self.tuning_blocks.append(nn.Sequential(
                default_conv(in_channels=channels, out_channels=channels,
                             kernel_size=3, stride=1, padding=1, bias=False, init_scale=0.1),
                Upsampler(default_conv, upscale, channels, bias=False, act=False),
            ))
            self.adaptive_alpha.append(nn.Sequential(
                default_Linear(64, 64, bias=False),
                default_Linear(64, 64, bias=False)
            ))

    def forward(self, x, alpha, number=0):
        input_alpha = self.control_alpha(alpha)
        # print(input_alpha.shape)
        tuning_f = self.tuning_blocks[number](x)
        # print(tuning_f.shape)
        ad_alpha = self.adaptive_alpha[number](input_alpha)
        ad_alpha = ad_alpha.view(-1, tuning_f.shape[1], 1, 1)
        # print(ad_alpha.shape)
        return tuning_f * ad_alpha, torch.ones_like(ad_alpha).cuda()-ad_alpha