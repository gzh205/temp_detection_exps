#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import UpsampleDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv, DownSampling


class DownsamplingBlock(nn.Module):
    def __init__(self,
                 depth=1.0,
                 width=1.0,
                 in_channels=256,
                 out_channels=512,
                 depthwise=False,
                 act="silu", ):
        super(DownsamplingBlock, self).__init__()
        Conv = DWConv if depthwise else BaseConv
        self.downsample = DownSampling(act=act)
        self.csp_layer = CSPLayer(
            int(out_channels * width),
            int(out_channels * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,

        )

    def forward(self, x, y):
        y_t = self.downsample(x)
        t = torch.cat([y, y_t], dim=1)
        t = self.csp_layer(t)
        return t


class UpsamplingBlock(nn.Module):
    def __init__(self,
                 depth=1.0,
                 width=1.0,
                 in_channels=256,
                 out_channels=256,
                 depthwise=False,
                 act="silu", ):
        super(UpsamplingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        Conv = DWConv if depthwise else BaseConv
        self.csp_layer = CSPLayer(
            int(in_channels * width),
            int(in_channels * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.conv = Conv(int(in_channels * width), int(out_channels * width), 1, 1, act=act)

    def forward(self, x, y):
        x_t = self.upsample(x)
        t = torch.cat([y, x_t], 1)
        t = self.csp_layer(t)
        t = self.conv(t)
        return t


class DPYOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark8", "dark7", "dark6", "dark2", "dark3", "dark4", "dark5"),
            in_channels=[16, 32, 64, 128, 256, 512, 1024],
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        self.in_features = in_features
        self.backbone = UpsampleDarknet(depth, width, out_features=in_features, depthwise=depthwise, act=act)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = Conv(in_channels[0], in_channels[1], 3, 1, 1, act=act)
        self.ConvReduceChannels = [
            BaseConv(int(in_channels[i] * width), int(in_channels[i - 1] * width), 1, 1, 1, act=act)
            for i in range(len(in_channels) - 1, 0, -1)
        ]
        self.up_conv_first = BaseConv(int(in_channels[len(in_channels) - 1] * width),
                                      out_channels=int(in_channels[len(in_channels) - 2] * width), ksize=1,
                                      stride=1, act=act)
        self.up_sample_last = nn.Upsample(scale_factor=2, mode="nearest")
        self.csp_layer_last = CSPLayer(int(in_channels[0] * 2 * width),
                                       int(in_channels[0] * width), round(3 * depth),
                                       False, depthwise=depthwise, act=act)
        self.up_samples = [
            UpsamplingBlock(depth, width, in_channels[i], in_channels[i - 1], depthwise=depthwise, act=act)
            for i in range(len(in_channels) - 1, 0, -1)]
        self.down_sample_first = DownSampling(act=act)
        self.down_samples = [
            DownsamplingBlock(depth, width, in_channels[i], in_channels[i],
                              depthwise=depthwise, act=act)
            for i in range(1, len(in_channels))]

    def forward(self, input):
        """
        Args:
            inputs: input images.
        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x6, x5, x4, x3, x2, x1, x0] = features

        # direct1
        f_out0 = self.ConvReduceChannels[0](x0)  # 1024->512
        f_out1 = self.up_samples[0](f_out0, x1)  # 512->256
        f_out1 = self.ConvReduceChannels[1](f_out1)
        f_out2 = self.up_samples[1](f_out1, x2)  # 256->128
        f_out2 = self.ConvReduceChannels[2](f_out2)
        f_out3 = self.up_samples[2](f_out2, x3)  # 128->64
        f_out3 = self.ConvReduceChannels[3](f_out3)
        f_out4 = self.up_samples[3](f_out3, x4)  # 64->32
        f_out4 = self.ConvReduceChannels[4](f_out4)
        f_out5 = self.up_samples[4](f_out4, x5)  # 32->16
        f_out5 = self.ConvReduceChannels[5](f_out5)
        fpn_up_out = self.up_sample_last(f_out5)
        fpn_up_out = torch.cat([fpn_up_out, x6], dim=1)
        f_out6 = self.csp_layer_last(fpn_up_out)

        # direct2
        f_out5 = self.down_samples[0](f_out6, f_out5)  # 16->32
        f_out4 = self.down_samples[1](f_out5, f_out4)  # 32->64
        f_out3 = self.down_samples[2](f_out4, f_out3)  # 64->128
        f_out2 = self.down_samples[3](f_out3, f_out2)  # 128->256
        f_out1 = self.down_samples[4](f_out2, f_out1)  # 256->512
        f_out0_t = self.down_sample_first(f_out1)  # 512->1024
        f_out0 = torch.cat([f_out0, f_out0_t], dim=1)

        outputs = (f_out6, f_out5, f_out4, f_out3, f_out2, f_out1, f_out0)
        return outputs
