# 该实验在Focus和第一次卷积之间添加了提取一阶和二阶微分算子的模块
# 替换了yolox.models.network_blocks.Focus模块
import os
from unittest.mock import patch

import torch
from torch import nn
from exps.example.yolox_voc import yolox_voc_s
from DifferentialOperator import FirstOrder, SecondOrder
import yolox.models.network_blocks
from exps.myexp import tools


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = yolox.models.network_blocks.BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)
        self.weight1 = torch.FloatTensor(FirstOrder.filter['scharr'])
        self.weight2 = torch.FloatTensor(SecondOrder.fliter['laplace'])
        print('init')

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        print('forward')
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        channels = x.shape[1]
        x1 = FirstOrder.conv(input=x, weights=self.weight1, channels=channels)
        x2 = SecondOrder.conv(input=x, weights=self.weight2, channels=channels)
        return self.conv(torch.cat([x, x1, x2], dim=1))


class Exp(yolox_voc_s.Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        tools.patch_class(yolox.models.network_blocks.Focus, Focus)
        # tools.patch_class(yolox.models.network_blocks, [Focus])

    def get_model(self):
        from yolox.models import YOLOX, YOLOXHead, YOLOPAFPN
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
