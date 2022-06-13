import os

import torch
from torch import nn

import yolox.models
from exps.example.yolox_voc import yolox_voc_s
from yolox.models import CSPDarknet
from yolox.models.network_blocks import get_activation, DWConv, BaseConv


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 dilation: int = 1,
                 act_type: str = 'silu'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            get_activation(act_type)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            get_activation(act_type)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            get_activation(act_type)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


class DilatedEncoder(nn.Module):
    """
    Dilated Encoder for YOLOF.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
          which are 1x1 conv + 3x3 conv
        - the dilated residual block
    """

    def __init__(self,
                 in_channels: int = 1024,
                 encoder_channels: int = 512,
                 block_mid_channels: int = 128,
                 num_residual_blocks: int = 4,
                 block_dilations=[2, 4, 6, 8],
                 width=1,
                 act_type='silu'
                 ):
        super(DilatedEncoder, self).__init__()
        # fmt: off
        # fmt: on
        assert len(block_dilations) == num_residual_blocks
        in_channels = in_channels * width
        encoder_channels = encoder_channels * width
        block_mid_channels = block_mid_channels * width
        # init
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, encoder_channels, kernel_size=1),
            nn.BatchNorm2d(encoder_channels),
            nn.Conv2d(encoder_channels, encoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels)
        )
        self.dilated_encoder_blocks = nn.Sequential(*[
            Bottleneck(
                encoder_channels,
                block_mid_channels,
                dilation=block_dilations[i],
                act_type=act_type
            ) for i in range(num_residual_blocks)
        ])

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out = self.input_conv(feature)
        return self.dilated_encoder_blocks(out)


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark5",),
            in_channels=[1024],
            depthwise=False,
            act='silu',
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, out_features=in_features, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_encoder = DilatedEncoder(width=width, act_type=act)

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
        x0 = features[0]

        return self.out_encoder(x0),


class Exp(yolox_voc_s.Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act = 'silu'

    def get_model(self):
        from yolox.models import YOLOX, YOLOXHead
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
